import os
import sys

from pathlib import Path


import argparse

import numpy as np
import matplotlib.pyplot as plt

import comet_ml
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import scipy.linalg as la
from src.dam_energies import *
from groups_datagen import (
    x_train,
    y_train,
    x_test,
    y_test,
    PreBatchedDataset,
    train_batched_data,
    test_batched_data,
)


API_KEY = Path(".comet_api").read_text().strip()

comet_ml.login()
# comet_ml.init(api_key=API_KEY) #will deprecate
workspace = Path(".comet_workspace").read_text().strip()

exp = comet_ml.Experiment(
    api_key=API_KEY,
    project_name="icd-gen",
    workspace=workspace,
    auto_metric_logging=True,  # default
)

from transformers.optimization import get_cosine_schedule_with_warmup

from baselines import (
    loss_if_predict_average,
    loss_if_predict_mostrecent,
    theory_linear_expected_error,  # TODO@DR: put this back
)
from data_util import (
    report_dataset_loss,
    report_dataset_psnr,
    data_train_test_split_linear,
    DatasetWrapper,
)
from vis_utils import vis_weights_kq_pv, vis_loss, vis_weights_grad_kq_pv

from models import *
from util import run_subdir_setup


torch.backends.cudnn.benchmark = True
if not torch.backends.mps.is_available():
    print("\nMPS device not found.")
    mps_device = None

if torch.backends.mps.is_available():
    device = torch.device("mps")
    mps_device = torch.device("mps")
    x = torch.ones(1, device=device)
    print("\nCheck M1 chip:", x)
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"
print("device selected:", device)


DIR_CURRENT = os.path.dirname(__file__)
DIR_PARENT = os.path.dirname(DIR_CURRENT)
sys.path.append(DIR_PARENT)

DIR_OUT = DIR_PARENT + os.sep + "output"
DIR_DATA = DIR_PARENT + os.sep + "input"
DIR_MODELS = DIR_PARENT + os.sep + "models"
DIR_RUNS = DIR_PARENT + os.sep + "runs"

for core_dir in [DIR_OUT, DIR_DATA, DIR_MODELS, DIR_RUNS]:
    if not os.path.exists(core_dir):
        os.mkdir(core_dir)

# Put in data_gen TODO@DR ADD New Ones; Ditch the Clustering
DATASET_CASES = {0: "Linear"}


def train_step(model, xs, ys, optimizer, loss_func):
    """the energy depends on some hyperparameters
    which have some simplest heuristic values hardcoded right now.

    """
    optimizer.zero_grad()
    # output = model(xs, ys)
    output_full, attn_arg = model(xs)
    output = output_full[:, :, -1]

    X = output_full[:, :, :-1]  # all but the last token
    # print("in train step device of xs ys", xs.device, ys.device)  # yeah mps so ok
    loss = loss_func(output, ys)

    # TODO@DR: I want also the energy and its gradient here
    loss.backward()
    optimizer.step()

    return (
        loss.detach().item(),
        output.detach(),
        output_full.detach(),
        attn_arg.detach(),
    )


def train(model, args):
    nn_model = args.training["nn_model"]

    if args.training["seed_torch"] is not None:
        torch.manual_seed(args.training["seed_torch"])

    ################################################################################
    # Data gen area in training
    ################################################################################

    ################################################################################
    # Model ID string
    ################################################################################

    # From the provided model class string, get the shorthand model name and the class definition
    nn_fpath = MODEL_CLASS_FROM_STR[nn_model]["alias"]
    MODEL_CLASS_FROM_STR[nn_model]["class"]

    epochs = args.training["epochs"]
    context_len = args.training["context_len"]
    dim_n = args.training["dim_n"]
    full_loss_sample_interval = args.training["full_loss_sample_interval"]
    batch_size = args.training["batch_size"]

    data_suffix = "grouped"
    opt_suffix = "adamw"

    model_fname = "%s_L%d_n%d_e%d_%s_%s" % (
        nn_fpath,
        context_len,
        dim_n,
        epochs,
        data_suffix,
        opt_suffix,
    )  # used as specialized label for model settings and run

    optimizer_lr = args.training["learning_rate"]  # 0.01  # 0.5

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_lr,
        betas=(
            0.9,
            0.98,
        ),  # DR: don't focus on all tunables here like the betas and the eps
        eps=1e-8,
        weight_decay=args.training["wd"],
    )

    ###################################################################
    # Setup io dict
    ################################################################################
    io_dict = run_subdir_setup(
        dir_runs=DIR_RUNS, run_subfolder=None, timedir_override=None, minimal_mode=False
    )

    state_path = os.path.join(DIR_RUNS, "state.pt")  # TODO@DR: not really sure where
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

    ################################################################################
    # Build or load data
    ################################################################################

    # specify training and testing datasets
    # Right now only for structured POC

    # just for save model
    # print(
    #     f"What dataset is wrapped for training********{x_train.shape} and label {y_train.shape}"
    # )

    if not args.training["batch_level_partitions"]:
        train_size = x_train.shape[0]
        print("train_size", train_size)
        train_dataset = DatasetWrapper(x_train, y_train)
        test_dataset = DatasetWrapper(x_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.training["nwork"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.training["nwork"],
        )
    else:
        # if groups at batch level - they get imported from group_datagen
        train_set = PreBatchedDataset(train_batched_data)  # already batched
        test_set = PreBatchedDataset(test_batched_data)

        train_size = len(train_batched_data)
        train_loader = DataLoader(
            train_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
        )

        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
        )

    # Freeze the projection layers
    # model.embedpatch.projection.weight.requires_grad = False
    # model.unembed.weight.requires_grad = False

    # see how many to train
    c1 = 0
    for param in model.parameters():
        if param.requires_grad:
            c1 += 1
    print(f"how many params require grad {c1}")

    if args.training["loss"] == "MSE":
        loss_func = nn.MSELoss()
    elif args.training["loss"] == "MAE":
        loss_func = nn.L1Loss()

    if args.training["scheduler_kwargs"]["choice"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, args.training["scheduler_kwargs"]["warmup"], epochs * train_size
        )
    ################################################################################
    # prep loss curves (x, y arrays)
    ################################################################################
    nbatches_per_epoch = np.ceil(x_train.shape[0] / batch_size)
    curve_x_losstrain_batch = (
        np.arange(1, epochs * nbatches_per_epoch + 1) / nbatches_per_epoch
    )
    curve_x_losstrain_epochs_avg = np.arange(1, epochs + 1)
    curve_x_losstest_interval = np.arange(
        0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch
    )

    # monitor the test error on the full test set every k batches (could be more/less than once per epoch)
    test_full_mse_loss = report_dataset_loss(
        model, loss_func, test_loader, "test", device
    )

    curve_y_losstrain_epochs_avg = []  # will append to this each epoch
    curve_y_losstrain_batch = (
        []
    )  # we begin tracking AFTER the first batch (could get loss nograd first batch here)

    curve_y_losstest_interval = [
        test_full_mse_loss
    ]  # will append to this each full_loss_sample_interval batches

    ################################################################################
    # train loop
    ################################################################################
    period_save_weights = args.training["period_save_weights"]

    for epoch in range(epochs):
        if epoch % period_save_weights == 0:
            # model_path = io_dict['dir_checkpoints'] + os.sep + 'model_e%d' % epoch + '.pth'
            torch.save(
                model.state_dict(),
                os.path.join(DIR_RUNS, f"model_{period_save_weights}.pt"),
            )

        running_loss_epoch = 0.0
        running_batch_counter = 0

        for i, data in enumerate(train_loader, 0):
            inputs, targets = data

            loss, output, output_full, attn_arg = train_step(
                model,
                inputs.to(device),
                targets.to(device),
                optimizer,
                loss_func,
            )

            curve_y_losstrain_batch.append(loss)
            running_loss_epoch += curve_y_losstrain_batch[-1]

            # (interval=4 batches) periodic inspection of test error
            if (running_batch_counter + 1) % full_loss_sample_interval == 0:
                loss_test = report_dataset_loss(
                    model, loss_func, test_loader, "test", device
                )

                curve_y_losstest_interval.append(loss_test)
                exp.log_metrics(
                    {"interval test loss": loss_test}, step=running_batch_counter
                )

            running_batch_counter += 1
            exp.log_metrics({"batch train loss": loss}, step=running_batch_counter)

            # TODO@DR: Reason through energies later

            # exp.log_metrics(
            #     {"batch energy 1 training example": energy.mean()},
            #     step=running_batch_counter,
            # )

        if args.training["scheduler_kwargs"]["choice"] == "cosine":
            scheduler.step()  # step the learning rate; if not cosine then no scheduler
            print("\tlast LR:", scheduler.get_last_lr())

        print("end epoch:", epoch, "====================")

        # Plot last activations (full_output) and attn_arg from the model
        print(
            "shape output full last from last batch as well as attn",
            output_full[-1, :, :].shape,
            attn_arg[-1, :, :].shape,
        )

        img_path = vis_weights_kq_pv(
            output_full[-1, :, :].detach().cpu().numpy(),
            attn_arg[-1, :, :].detach().cpu().numpy(),
            titlemod=f"repres. final and intermed repres. last in epoch {epoch}",
            dir_out=io_dict["dir_vis"],
            fname="representations final per epoch",
            flag_show=args.training["flag_vis_weights"],
        )

        img = Image.open(img_path)

        exp.log_image(
            image_data=img,
            name=f"attn_repres{epoch}.png",
            image_format="png",
            step=0,
        )

        ep_loss = running_loss_epoch / running_batch_counter

        # for name, param in model.named_parameters():
        #     # print(f"param is {param} and grad is {param.grad}")
        #     if param.grad is not None:
        #         print(
        #             f"In epoch {epoch} shape of {name} is {param.size()} and gradient is {param.grad.size()}"
        #         )

        curve_y_losstrain_epochs_avg.append(ep_loss)  # DR: keep for paper-like vis

        exp.log_metrics({"epoch avg train loss": ep_loss}, epoch=epoch)

    print("Finished Training")

    ################################################################################
    # Save model
    ################################################################################

    # save a copy of final model as 'model_final.pth'
    model_path = io_dict["dir_checkpoints"] + os.sep + "model_final" + ".pth"
    torch.save(
        model.state_dict(), io_dict["dir_checkpoints"] + os.sep + "model_final" + ".pth"
    )
    print("\nModel checkpoint saved to", model_path)

    report_dataset_loss(model, loss_func, train_loader, "train", device)
    report_dataset_loss(model, loss_func, test_loader, "test", device)
    report_dataset_psnr(model, loss_func, test_loader, "test", device)

    print("curve_x_losstrain_epochs_avg", curve_x_losstrain_epochs_avg)
    print("curve_y_losstrain_epochs_avg", curve_y_losstrain_epochs_avg, "\n")

    print("curve_x_losstest_interval", curve_x_losstest_interval)
    print("curve_y_losstest_interval", curve_y_losstest_interval, "\n")

    return (
        model,
        model_fname,
        io_dict,
        train_loader,
        test_loader,
        x_train,
        y_train,
        x_test,
        y_test,
    )


def main(args):
    model = TransformerModelV2(
        context_length=args.training["context_len"],
        dim_input=args.training["dim_n"],
        add_frozen_kernel=True,
        backbone="ViT",
    )
    print(model)

    model.to(device)
    model.train()
    (
        net,
        model_fname,
        io_dict,
        train_loader,
        test_loader,
        x_train,
        y_train,
        x_test,
        y_test,
    ) = train(model, args)

    print(f"what did train on {x_train.shape}")

    learned_W_KQ = net.W_KQ.detach().cpu().numpy()
    learned_W_PV = net.W_PV.detach().cpu().numpy()

    # TODO@DR: Also visualize grads and activations for full understanding
    learned_W_KQ_grad = net.W_KQ.grad.detach().cpu().numpy()
    learned_W_PV_grad = net.W_PV.grad.detach().cpu().numpy()

    # Q, R, perm = la.qr(learned_W_KQ, pivoting=True)
    # print(f"What is R for learned_W_KQ rank-reveal decomp: {R}")
    # # TODO@DR: Do some rank analysis

    rank_wkq = np.linalg.matrix_rank(learned_W_KQ)
    print(f"rank of learned_W_KQ is {rank_wkq}")  # 16
    rank_wpv = np.linalg.matrix_rank(learned_W_PV)
    print(f"rank of learned_W_PV is {rank_wpv}")  # 16

    rank_wkq = np.linalg.matrix_rank(learned_W_KQ_grad)
    print(f"rank of learned_W_KQ grad last is {rank_wkq}")  # 4
    rank_wpv = np.linalg.matrix_rank(learned_W_PV_grad)
    print(f"rank of learned_W_PV grad last is {rank_wpv}")  # 4
    # TODO@DR - is there something about the rank of the grad mat
    # at the local optimums?

    img_path = vis_weights_kq_pv(
        learned_W_KQ,
        learned_W_PV,
        titlemod=r"$\theta$ final",
        dir_out=io_dict["dir_vis"],
        fname="weights_final",
        flag_show=args.training["flag_vis_weights"],
    )

    img = Image.open(img_path)

    exp.log_image(
        image_data=img,
        name="attn_weigths.png",
        image_format="png",
        step=0,
    )

    # plot grads
    img_path = vis_weights_grad_kq_pv(
        learned_W_KQ_grad,
        learned_W_PV_grad,
        titlemod=r"$\theta$ grad",
        dir_out=io_dict["dir_vis"],
        fname="weight_mats_grads",
        flag_show=args.training["flag_vis_grad"],
    )

    img = Image.open(img_path)

    exp.log_image(
        image_data=img,
        name="attn_weigths_grads.png",
        image_format="png",
        step=0,
    )

    # Log groups from structured datagen
    img = Image.open("groups_built_in_datagen.png")

    exp.log_image(
        image_data=img,
        name=f"groups_built_in_datagen.png",
        image_format="png",
        step=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for the icd-gen task.")
    parser.add_argument("--config-file", help="Path to YAML config file")

    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
            parser.set_defaults(**config)
        # Reload arguments to apply YAML values
        args = parser.parse_args()

    # print(f"Running with: {args}")
    # exp.log_parameters(args)

    with open(os.path.join(DIR_OUT, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
    exp.end()

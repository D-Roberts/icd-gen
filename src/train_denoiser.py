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
from energies import *
from groups_datagen import x_train, y_train, x_test, y_test

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
    # loss_if_predict_zero,
    loss_if_predict_average,
    loss_if_predict_mostrecent,
    # loss_if_predict_linalg,
    # loss_if_predict_linalg_shrunken, # TODO@DR: put this back
    # theory_linear_expected_error, # TODO@DR: put this back
)
from data_util import report_dataset_loss, data_train_test_split_linear, DatasetWrapper
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
DATASET_CASES = {0: "Linear", 2: "Manifold (sphere)"}

# TODO@DR hooks when I have more layers POC
# activations = {}

# def get_activation(name):
#     def hook(model, input, output):
#         activations[name] = output.detach()
#     return hook
# model.layer_name.register_forward_hook(get_activation('attn_output'))


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

    # I want also the energy and its gradient here
    # TODO@DR: these are for one training example in batch until I refactor
    # the energy and grad_energy to be vectorized and torchified
    one_eg_q = output[0, :].squeeze().detach().cpu().numpy()
    one_eg_X = X[0, :, :].squeeze().detach().cpu().numpy()

    en = energy(q=one_eg_q, X=one_eg_X, c_lambda=1.0, beta=1.0, c_k=1.0)
    grad_en = grad_energy(q=one_eg_q, X=one_eg_X, c_lambda=1.0, beta=1.0, c_k=1.0)

    loss.backward()
    optimizer.step()
    return (
        loss.detach().item(),
        output.detach(),
        output_full.detach(),
        attn_arg.detach(),
        en,
        grad_en,
    )


def train(model, args):
    nn_model = args.training["nn_model"]

    if nn_model is None:
        # select a model class:
        nn_model = "TransformerModelV1noresOmitLast"

    if args.training["seed_torch"] is not None:
        torch.manual_seed(args.training["seed_torch"])

    ################################################################################
    # Data gen area in training
    ################################################################################

    assert args.data["datagen_case"] in DATASET_CASES.keys()
    datagen_choice = "linear"  # or manifold for now

    # # asserts for Linear case
    assert args.DATAGEN_GLOBALS[datagen_choice]["style_subspace_dimensions"] in [
        "random"
    ] or isinstance(args.DATAGEN_GLOBALS["linear"]["style_subspace_dimensions"], int)
    assert args.DATAGEN_GLOBALS[datagen_choice]["style_corruption_orthog"] is False
    assert args.DATAGEN_GLOBALS[datagen_choice]["style_origin_subspace"] is True

    ###datagen_choice = 1   # {0, 1, 2} -> {linear, clusters, manifold}
    ["linear"][0]

    base_kwargs = dict(
        context_len=args.training["context_len"],
        dim_n=args.training["dim_n"],
        num_W_in_dataset=args.training["num_W_in_dataset"],
        context_examples_per_W=args.training["context_examples_per_W"],
        samples_per_context_example=args.training["samples_per_context_example"],
        test_ratio=args.training["test_ratio"],
        verbose=True,
        as_torch=True,
        savez_fname=None,
        seed=args.data["datagen_seed"],
        style_subspace_dimensions=args.DATAGEN_GLOBALS[datagen_choice][
            "style_subspace_dimensions"
        ],
        style_origin_subspace=args.DATAGEN_GLOBALS[datagen_choice][
            "style_origin_subspace"
        ],
        style_corruption_orthog=args.DATAGEN_GLOBALS[datagen_choice][
            "style_corruption_orthog"
        ],
        sigma2_corruption=args.DATAGEN_GLOBALS[datagen_choice]["sigma2_corruption"],
    )

    # Focus on linear data case for now

    if datagen_choice == "linear":
        data_train_test_split_fncall = data_train_test_split_linear

        # specific to this case but used in function calls below
        linear_sigma2_pure_context = args.DATAGEN_GLOBALS[datagen_choice][
            "sigma2_pure_context"
        ]
        sigma2_pure_context = linear_sigma2_pure_context  # alias

        data_suffix = "case%s_s2z%.2f_s2n%.2f_ortho%d_origin%d_d-%s" % (
            datagen_choice,
            args.DATAGEN_GLOBALS["linear"]["sigma2_pure_context"],
            args.DATAGEN_GLOBALS["linear"]["sigma2_corruption"],
            args.DATAGEN_GLOBALS["linear"]["style_corruption_orthog"],
            args.DATAGEN_GLOBALS["linear"]["style_origin_subspace"],
            args.DATAGEN_GLOBALS["linear"]["style_subspace_dimensions"],
        )

    # shorthand aliases used below
    sigma2_corruption = args.DATAGEN_GLOBALS["linear"]["sigma2_corruption"]  # float
    style_corruption_orthog = args.DATAGEN_GLOBALS["linear"][
        "style_corruption_orthog"
    ]  # True/False (orthog OR ball)
    style_origin_subspace = args.DATAGEN_GLOBALS["linear"][
        "style_origin_subspace"
    ]  # True/False
    style_subspace_dimensions = args.DATAGEN_GLOBALS["linear"][
        "style_subspace_dimensions"
    ]  # int or 'random' (or just 'full' in clustering case)

    """ DR: this is from ic denoise repo
    HERE - linear: How many samples per in-context subspace? - We use option (A) (explained below) throughout
    # (X, y) samples style A
    num_W_in_dataset = train_plus_test_size
    context_examples_per_W = 1
    samples_per_context_example = 1
    """
    num_W_in_dataset = args.training["num_W_in_dataset"]
    context_examples_per_W = args.training["context_examples_per_W"]
    samples_per_context_example = args.training["samples_per_context_example"]
    assert context_examples_per_W == 1 and samples_per_context_example == 1
    args.training[
        "train_plus_test_size"
    ] == context_examples_per_W * num_W_in_dataset * samples_per_context_example  # sanity check
    # this means train_plus_test_size == context_examples_per_W   i.e. we use option (A) throughout

    test_ratio = args.training[
        "test_ratio"
    ]  # 0.2 means 1000 -> 800, 200 samples for train/test
    args.data["datagen_seed"]
    ################################################################################
    # Model ID string
    ################################################################################
    # Data parameters
    if args.training["flag_save_dataset"]:
        print(
            "Warning - flag_save_dataset - 1000 samples with n=32 still gives 60 MB, relatively big)"
        )

    # From the provided model class string, get the shorthand model name and the class definition
    nn_fpath = MODEL_CLASS_FROM_STR[nn_model]["alias"]
    MODEL_CLASS_FROM_STR[nn_model]["class"]

    opt_suffix = "adamw"

    epochs = args.training["epochs"]
    context_len = args.training["context_len"]
    dim_n = args.training["dim_n"]
    full_loss_sample_interval = args.training["full_loss_sample_interval"]
    batch_size = args.training["batch_size"]

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

    # Output settings for visualization after training
    skip_PCA_heuristic_slow = args.training["skip_PCA_heuristic_slow"]

    ################################################################################
    # Setup io dict
    ################################################################################
    io_dict = run_subdir_setup(
        dir_runs=DIR_RUNS, run_subfolder=None, timedir_override=None, minimal_mode=False
    )

    if args.training["flag_save_dataset"]:
        (
            io_dict["dir_base"] + os.sep + "training_dataset_split.npz"
        )  # if none, do not save
        # datagen_kwargs['savez_fname'] = dataset_savez_fname
    else:
        pass
        # datagen_kwargs['savez_fname'] = dataset_savez_fname

    # this is from icl
    state_path = os.path.join(DIR_RUNS, "state.pt")  # TODO@DR: not really sure where
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

    # (
    #     x_train,
    #     y_train,
    #     x_test,
    #     y_test,
    #     train_data_subspaces,
    #     test_data_subspaces,
    # ) = data_train_test_split_linear(
    #     **base_kwargs,
    #     sigma2_pure_context=args.DATAGEN_GLOBALS[datagen_choice]["sigma2_pure_context"],
    # )

    # print(
    #     "in train denoiser, shape and device of x_train", x_train.shape, x_train.device
    # )

    ################################################################################
    # Build or load data
    ################################################################################

    # keep for the linear manifold ablations
    # (
    #     x_train,
    #     y_train,
    #     x_test,
    #     y_test,
    #     train_data_subspaces,
    #     test_data_subspaces,
    # ) = data_train_test_split_fncall(**base_kwargs)

    # print("x_train.shape", x_train.shape)

    # specify training and testing datasets
    train_size = x_train.shape[0]
    print("train_size", train_size)
    print(
        "int(train_plus_test_size * (1 - test_ratio))",
        int(args.training["train_plus_test_size"] * (1 - test_ratio)),
    )
    assert train_size == int(
        args.training["train_plus_test_size"] * (1 - test_ratio)
    )  # sanity check

    data_suffix += "_tx%d_xpw%d_spx%d" % (
        train_size,
        context_examples_per_W,
        samples_per_context_example,
    )

    print(f"What is wrapped for training********{x_train.shape}")

    train_dataset = DatasetWrapper(x_train, y_train)
    test_dataset = DatasetWrapper(x_test, y_test)

    ["\toptimizer_lr, %.2e" % optimizer_lr]
    opt_suffix = "adamw%.1e" % optimizer_lr  # appended to fname

    if args.training["scheduler_kwargs"]["choice"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, args.training["scheduler_kwargs"]["warmup"], epochs * train_size
        )

    nwork = 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork
    )  # TODO@DR: is this correct to shuffle in test?

    # inspect network class
    params = list(model.parameters())
    print("\nNum of params matrices to train:", len(params))
    print("\tparams[0].size():", params[0].size())

    loss_func = nn.MSELoss()

    ################################################################################
    # prep loss curves (x, y arrays)
    ################################################################################
    nbatches_per_epoch = np.ceil(x_train.shape[0] / batch_size)
    curve_x_losstrain_batch = (
        np.arange(1, epochs * nbatches_per_epoch + 1) / nbatches_per_epoch
    )
    curve_x_losstrain_epochs_avg = np.arange(
        1, epochs + 1
    )  # average over batches in the epoch (fast but inaccurate estimate)
    curve_x_losstest_interval = np.arange(
        0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch
    )
    # curve_x_losstrain_interval = np.arange(
    #     0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch
    # )

    # monitor the train error on the full test set every k batches (could be more/less than once per epoch)
    train_full_mse_loss = report_dataset_loss(
        model, loss_func, train_loader, "train", device
    )
    # monitor the test error on the full test set every k batches (could be more/less than once per epoch)
    test_full_mse_loss = report_dataset_loss(
        model, loss_func, test_loader, "test", device
    )

    curve_y_losstrain_epochs_avg = []  # will append to this each epoch
    curve_y_losstrain_batch = (
        []
    )  # we begin tracking AFTER the first batch (could get loss nograd first batch here)
    batch_energies = []
    batch_energy_grads = []

    curve_y_losstest_interval = [
        test_full_mse_loss
    ]  # will append to this each full_loss_sample_interval batches
    # curve_y_losstrain_interval = [train_full_mse_loss]  # will append to this each epoch

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
            # torch.save(model.state_dict(), model_path)

        running_loss_epoch = 0.0
        # running_loss_mesoscale = 0.0
        running_batch_counter = 0
        print("\nepoch:", epoch)

        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            # print(
            #     "inputs shape and device", inputs.shape, inputs.device
            # )  # (batch size, x token dimension, context len)
            # print(
            #     "targets shape and device", targets.shape, targets.device
            # )  # (batch size, n dim of last token)

            loss, output, output_full, attn_arg, energy, energy_grad = train_step(
                model,
                inputs.to(device),
                targets.to(device),
                optimizer,
                loss_func,
            )

            curve_y_losstrain_batch.append(loss)

            # batch_energies.append(energy)
            # batch_energy_grads.append(energy_grad)

            # print statistics
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
            exp.log_metrics(
                {"batch energy 1 training example": energy.mean()},
                step=running_batch_counter,
            )
            exp.log_metrics(
                {"batch energy grad 1 training example": energy_grad.mean()},
                step=running_batch_counter,
            )

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
            titlemod=f"activations final and attnarg last in epoch {epoch}",
            dir_out=io_dict["dir_vis"],
            fname="activations final per epoch",
            flag_show=args.training["flag_vis_weights"],
        )

        img = Image.open(img_path)

        exp.log_image(
            image_data=img,
            name=f"attn_activations{epoch}.png",
            image_format="png",
            step=0,
        )

        ep_loss = running_loss_epoch / running_batch_counter

        # for name, param in model.named_parameters():
        #     # print(f"param is {param} and grad is {param.grad}")
        #     if param.grad is not None:
        #         print(f"In epoch {epoch} shape of {name} is {param.size()} and gradient is {param.grad.size()}")

        curve_y_losstrain_epochs_avg.append(ep_loss)  # DR: keep for paper-like vis

        exp.log_metrics({"epoch avg train loss": ep_loss}, epoch=epoch)

    print("Finished Training")

    ################################################################################
    # Save model
    ################################################################################
    # save a copy of final model using detailed fname label
    model_path = io_dict["dir_base"] + os.sep + model_fname + ".pth"
    torch.save(model.state_dict(), model_path)
    # save a copy of final model as 'model_final.pth'
    model_path = io_dict["dir_checkpoints"] + os.sep + "model_final" + ".pth"
    torch.save(
        model.state_dict(), io_dict["dir_checkpoints"] + os.sep + "model_final" + ".pth"
    )
    print("\nModel checkpoint saved to", model_path)

    report_dataset_loss(model, loss_func, train_loader, "train", device)
    report_dataset_loss(model, loss_func, test_loader, "test", device)

    print("curve_x_losstrain_epochs_avg", curve_x_losstrain_epochs_avg)
    print("curve_y_losstrain_epochs_avg", curve_y_losstrain_epochs_avg, "\n")

    print("curve_x_losstest_interval", curve_x_losstest_interval)
    print("curve_y_losstest_interval", curve_y_losstest_interval, "\n")

    # skip the saving of info to txt - I have the yaml dump TODO@DR add to that
    ################################################################################
    # Plot loss dynamics against simple benchmarks
    ################################################################################

    loss_vals_dict = {
        "loss_train_batch": dict(
            x=curve_x_losstrain_batch,
            y=curve_y_losstrain_batch,
            label="train (one batch)",
            fname="curve_loss_train_batch",
            pltkwargs=dict(
                linestyle="--",
                marker="o",
                color="b",
                markersize=4,
                markerfacecolor="None",
                alpha=0.3,
            ),
        ),
        "loss_train_epoch_avg": dict(
            x=curve_x_losstrain_epochs_avg,
            y=curve_y_losstrain_epochs_avg,
            label="train (epoch moving avg)",
            fname="curve_loss_train_epoch_avg",
            pltkwargs=dict(linestyle="--", marker="o", color="b", markersize=4),
        ),
        "loss_test_interval": dict(
            x=curve_x_losstest_interval,
            y=curve_y_losstest_interval,
            label="test (full)",
            fname="curve_loss_test_interval",
            pltkwargs=dict(linestyle="-", marker="o", color="r"),
        ),
    }
    print("Compare to baselines:")

    # HAVE V2 here as baseline one-layer TODO@DR must train on same
    simple_B_mse_on_train = loss_if_predict_mostrecent(loss_func, train_loader, "train")
    simple_B_mse_on_test = loss_if_predict_mostrecent(loss_func, test_loader, "test")
    simple_C_mse_on_train = loss_if_predict_average(loss_func, train_loader, "train")
    simple_C_mse_on_test = loss_if_predict_average(loss_func, test_loader, "test")

    # add core baselines to loss_vals_dict (will also add datagen-case-specific ones later)
    loss_vals_dict["baselines"] = {
        # #TODO@DR: predict with the baseline one-layer V2 softmax instead
        # otherwise - leave only the simplest models that would work on
        # multiple underlying distrib as baselines - average and most recent.
        "loss_if_predict_mostrecent": dict(
            alias="recent",
            label=r"predict $x_{k-1}$",
            val_train=simple_B_mse_on_train,
            val_test=simple_B_mse_on_test,
            pltkwargs=dict(color="green"),
        ),
        "loss_if_predict_average": dict(
            alias="mean",  # TODO@DR: change names; keep the most recent and average as simple predictors, together with the one-layer linearized
            # softmax
            label=r"predict mean",
            val_train=simple_C_mse_on_train,
            val_test=simple_C_mse_on_test,
            pltkwargs=dict(color="orange"),
        ),
    }

    return (
        model,
        model_fname,
        io_dict,
        loss_vals_dict,
        train_loader,
        test_loader,
        x_train,
        y_train,
        x_test,
        y_test,
        train_data_subspaces,
        test_data_subspaces,
    )


def main(args):
    if args.model["type"] in {"gpt2"}:
        model = build_model(args.model)
    else:
        model = TransformerModelV2(args.training["context_len"], args.training["dim_n"])

    model.to(device)
    model.train()
    (
        net,
        model_fname,
        io_dict,
        loss_vals_dict,
        train_loader,
        test_loader,
        x_train,
        y_train,
        x_test,
        y_test,
        train_data_subspaces,
        test_data_subspaces,
    ) = train(model, args)

    print(f"what did train on {x_train.shape}")

    if args.model["type"] not in {"gpt2"}:
        learned_W_KQ = net.W_KQ.detach().cpu().numpy()
        learned_W_PV = net.W_PV.detach().cpu().numpy()

        # TODO@DR: Also visualize grads and activations for full understanding
        learned_W_KQ_grad = net.W_KQ.grad.detach().cpu().numpy()
        learned_W_PV_grad = net.W_PV.grad.detach().cpu().numpy()

        # Q, R, perm = la.qr(learned_W_KQ, pivoting=True)
        # print(f"What is R for learned_W_KQ rank-reveal decomp: {R}")
        # # TODO@DR: Do some rank analysis - (the papers with the ranks)

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

        # plot with the baselines
        # TODO@DR: this broke; fix

        # img_path_loss_baselines = vis_loss(
        #     loss_vals_dict=loss_vals_dict,
        #     titlemod="with baselines",
        #     dir_out=io_dict["dir_vis"],
        #     fname="Train Test Losses and Baselines",
        #     flag_show=args.training["flag_vis_weights"],
        # )
        # img = Image.open(img_path_loss_baselines)

        # exp.log_image(
        #     image_data=img,
        #     name="loss baselines.png",
        #     image_format="png",
        #     step=0,
        # )

        # Log groups from structured datagen
        img = Image.open("see_groups.png")

        exp.log_image(
            image_data=img,
            name=f"groups.png",
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

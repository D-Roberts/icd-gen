import os
import sys

from pathlib import Path

from random import randint
import uuid

import argparse
import wandb
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax as scipy_softmax

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import torch.optim as optim

import comet_ml
API_KEY = Path(".comet_api").read_text().strip()
comet_ml.init(api_key=API_KEY)

from transformers.optimization import get_cosine_schedule_with_warmup

from baselines import (
    loss_if_predict_zero,
    loss_if_predict_average,
    loss_if_predict_mostrecent,
    loss_if_predict_linalg,
    loss_if_predict_linalg_shrunken,
    theory_linear_expected_error,
)
from data_util import report_dataset_loss, data_train_test_split_linear, DatasetWrapper
from vis_utils import vis_weights_kq_pv

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
DATASET_CASES = {0: "Linear", 1: "Clustering", 2: "Manifold (sphere)"}


class DatasetWrapper(Dataset):
    """
    (relic): currently, there is a "remainder" batch at the end, with size smaller than batch_size -- could discard it
    """

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

        self.dim_n = self.x.size()[1]
        self.context_length = self.x.size()[2]

    # Mandatory: Get input pair for training
    def __getitem__(self, idx):
        return self.x[idx, :, :], self.y[idx, :]

    # Mandatory: Number of elements in dataset (i.e. size of batch dimension 0)
    def __len__(self):
        X_len = self.x.size()[0]
        return X_len

    # This function is not needed
    def plot(self):
        print("not implemented")
        return


# Leave this as is from icl; see that I use the scheduler TODO@DR
def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    # output = model(xs, ys)
    output = model(xs)
    # print("in train step device of xs ys", xs.device, ys.device)  # yeah mps so ok
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


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
    datagen_label = ["linear"][0]

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

    # elif datagen_choice == "manifold": #TODO: later

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

    """ HERE - linear: How many samples per in-context subspace? - We use option (A) (explained below) throughout 
    # (X, y) samples style A
    num_W_in_dataset = train_plus_test_size
    context_examples_per_W = 1
    samples_per_context_example = 1
    """
    num_W_in_dataset = args.training[
        "num_W_in_dataset"
    ]  # we assert context_examples_per_W = 1, samples_per_context_example = 1
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
    seed_dataset = args.data["datagen_seed"]
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
    nn_class = MODEL_CLASS_FROM_STR[nn_model]["class"]

    opt_suffix = "adam"

    epochs = args.training["epochs"]
    context_len = args.training["context_len"]
    dim_n = args.training["dim_n"]
    full_loss_sample_interval = args.training["full_loss_sample_interval"]
    batch_size = args.training["batch_size"]

    optimizer_choice = args.training["optimizer_choice"]  # DR: just one.

    model_fname = "%s_L%d_n%d_e%d_%s_%s" % (
        nn_fpath,
        context_len,
        dim_n,
        epochs,
        data_suffix,
        opt_suffix,
    )  # used as specialized label for model settings and run

    if optimizer_choice == "adamw":
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
    else:
        assert optimizer_choice == "adam"
        optimizer_lr = args.training["learning_rate"]  # default: 1e-2

    # Output settings for visualization after training
    skip_PCA_heuristic_slow = args.training["skip_PCA_heuristic_slow"]

    ################################################################################
    # Setup io dict
    ################################################################################
    io_dict = run_subdir_setup(
        dir_runs=DIR_RUNS, run_subfolder=None, timedir_override=None, minimal_mode=False
    )

    if args.training["flag_save_dataset"]:
        dataset_savez_fname = (
            io_dict["dir_base"] + os.sep + "training_dataset_split.npz"
        )  # if none, do not save
        # datagen_kwargs['savez_fname'] = dataset_savez_fname
    else:
        dataset_savez_fname = None
        # datagen_kwargs['savez_fname'] = dataset_savez_fname

    # this is from icl
    state_path = os.path.join(DIR_RUNS, "state.pt")  # TODO@DR: not really sure where
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

    # pbar = tqdm(range(starting_step, args.training["train_steps"])) TODO: put this back later

    x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = (
        data_train_test_split_linear(
            **base_kwargs,
            sigma2_pure_context=args.DATAGEN_GLOBALS[datagen_choice][
                "sigma2_pure_context"
            ],
        )
    )

    print(
        "in train denoiser, shape and device of x_train", x_train.shape, x_train.device
    )

    ################################################################################
    # Build or load data
    ################################################################################
    restart_nn_instance = None
    restart_dataset = None

    if args.training["restart_dataset"] is not None:
        x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = (
            restart_dataset
        )
        print("x_train.shape", x_train.shape)

        # specify training and testing datasets
        train_size = x_train.shape[0]
        assert train_size == int(
            args.training["train_plus_test_size"] * (1 - test_ratio)
        )  # sanity check

        # fname suffix for io
        data_suffix = "RESTART-SAME-DATASET"  # appended to fname
    else:
        x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = (
            data_train_test_split_fncall(**base_kwargs)
        )

        print("x_train.shape", x_train.shape)

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

    train_dataset = DatasetWrapper(x_train, y_train)
    test_dataset = DatasetWrapper(x_test, y_test)

    runinfo_optimizer_lines = ["\toptimizer_lr, %.2e" % optimizer_lr]
    opt_suffix = "adam%.1e" % optimizer_lr  # appended to fname

    # I'm not using sgd with momentum

    if args.training["scheduler_kwargs"] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 10, epochs * train_size
        )  # TODO@DR put warmup steps in yaml config; ditch the other schedules - either cosine or None

    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epochs + 1], gamma=1.0
        )  # so this should be None

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
    curve_x_losstrain_interval = np.arange(
        0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch
    )

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
    curve_y_losstest_interval = [
        test_full_mse_loss
    ]  # will append to this each full_loss_sample_interval batches
    curve_y_losstrain_interval = [train_full_mse_loss]  # will append to this each epoch

    ################################################################################
    # train loop
    ################################################################################
    period_save_weights = args.training["period_save_weights"]
    count = 1

    for epoch in range(epochs):

        if epoch % period_save_weights == 0:
            # model_path = io_dict['dir_checkpoints'] + os.sep + 'model_e%d' % epoch + '.pth'
            torch.save(
                model.state_dict(),
                os.path.join(DIR_RUNS, f"model_{period_save_weights}.pt"),
            )
            # torch.save(model.state_dict(), model_path)

        running_loss_epoch = 0.0
        running_loss_mesoscale = 0.0
        running_batch_counter = 0
        print("\nepoch:", epoch)

        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            print(
                "inputs shape and device", inputs.shape, inputs.device
            )  # (batch size, x token dimension, context len)
            print(
                "targets shape and device", targets.shape, targets.device
            )  # (batch size, n dim of last token)

            # train step from icl
            loss, output = train_step(
                model, inputs.to(device), targets.to(device), optimizer, loss_func
            )
            # print("loss ", loss)

            curve_y_losstrain_batch.append(loss)

            # print statistics
            running_loss_epoch += curve_y_losstrain_batch[-1]  # was [count]
            running_loss_mesoscale += curve_y_losstrain_batch[-1]  # was [count]
            # (slow) periodic inspection of test error
            if (
                count % full_loss_sample_interval == 0
            ):  # report it every "full_loss_sample_interval" batches
                print(
                    "Epoch: %d, batch: %4d, loss (avg): %.2e"
                    % (epoch, count, running_loss_mesoscale / full_loss_sample_interval)
                )
                print(
                    "running_loss_mesoscale, full_loss_sample_interval, count |",
                    running_loss_mesoscale,
                    full_loss_sample_interval,
                    count,
                )

                loss_test = report_dataset_loss(
                    model, loss_func, test_loader, "test", device
                )
                curve_y_losstest_interval.append(loss_test)

                loss_train = report_dataset_loss(
                    model, loss_func, train_loader, "train", device
                )
                curve_y_losstrain_interval.append(loss_train)

                running_loss_mesoscale = 0.0

            count += 1  # count tracks number of batches which have been trained over (at this point)
            running_batch_counter += 1

        scheduler.step()  # step the learning rate sschedulecheduler
        print("\tlast LR:", scheduler.get_last_lr())
        print("end epoch:", epoch, "====================")
        curve_y_losstrain_epochs_avg.append(running_loss_epoch / running_batch_counter)

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

    train_loss_end = report_dataset_loss(
        model, loss_func, train_loader, "train", device
    )
    test_loss_end = report_dataset_loss(model, loss_func, test_loader, "test", device)

    print("curve_x_losstrain_epochs_avg", curve_x_losstrain_epochs_avg)
    print("curve_y_losstrain_epochs_avg", curve_y_losstrain_epochs_avg, "\n")

    print("curve_x_losstrain_batch", curve_x_losstrain_batch)
    print("curve_y_losstrain_batch", curve_y_losstrain_batch, "\n")

    print("curve_x_lossrain_interval", curve_x_losstrain_interval)
    print("curve_y_losstrain_interval", curve_y_losstrain_interval, "\n")

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
        "loss_train_interval": dict(
            x=curve_x_losstrain_interval,
            y=curve_y_losstrain_interval,
            label="train (full)",
            fname="curve_loss_train_interval",
            pltkwargs=dict(linestyle="-", marker="o", color="b"),
        ),
        "loss_test_interval": dict(
            x=curve_x_losstest_interval,
            y=curve_y_losstest_interval,
            label="test (full)",
            fname="curve_loss_test_interval",
            pltkwargs=dict(linestyle="-", marker="o", color="r"),
        ),
    }
    print("Compare to null performance and lin.alg. baselines:")
    dumb_A_mse_on_train = loss_if_predict_zero(loss_func, train_loader, "train")
    dumb_A_mse_on_test = loss_if_predict_zero(loss_func, test_loader, "test")
    dumb_B_mse_on_train = loss_if_predict_mostrecent(loss_func, train_loader, "train")
    dumb_B_mse_on_test = loss_if_predict_mostrecent(loss_func, test_loader, "test")
    dumb_C_mse_on_train = loss_if_predict_average(loss_func, train_loader, "train")
    dumb_C_mse_on_test = loss_if_predict_average(loss_func, test_loader, "test")

    # add core baselines to loss_vals_dict (will also add datagen-case-specific ones later)
    loss_vals_dict["baselines"] = {
        "loss_if_predict_zero": dict(
            alias="dumb_A",
            label=r"guess $0$",
            val_train=dumb_A_mse_on_train,
            val_test=dumb_A_mse_on_test,
            pltkwargs=dict(color="grey"),
        ),
        "loss_if_predict_mostrecent": dict(
            alias="dumb_B",
            label=r"guess $x_{k-1}$",
            val_train=dumb_B_mse_on_train,
            val_test=dumb_B_mse_on_test,
            pltkwargs=dict(color="green"),
        ),
        "loss_if_predict_average": dict(
            alias="dumb_C",  # TODO@DR: change names; keep the most recent and average as simple predictors, together with the one-layer linearized
            # softmax
            label=r"guess mean",
            val_train=dumb_C_mse_on_train,
            val_test=dumb_C_mse_on_test,
            pltkwargs=dict(color="orange"),
        ),
    }
    # the following heuristics baselines are specific to case 0: Linear subspaces
    if datagen_choice == "linear":
        if not skip_PCA_heuristic_slow:
            print("Warning: not skip_PCA_heuristic_slow; slow lin.alg. step...")
            heuristic_mse_on_train = loss_if_predict_linalg(
                loss_func, train_loader, "train"
            )
            heuristic_mse_on_test = loss_if_predict_linalg(
                loss_func, test_loader, "test"
            )

            loss_vals_dict["baselines"]["loss_if_predict_linalg"] = dict(
                alias="heuristic_proj",
                label=r"$P \tilde x$",
                val_train=heuristic_mse_on_train,
                val_test=heuristic_mse_on_test,
                pltkwargs=dict(color="black"),
            )

            # also compute shrunken predictor
            # - we assume proper subspace through origin
            # - we assume it is iid gaussian ball corruption (not orthogonal to W)
            if (style_origin_subspace) and (
                not style_corruption_orthog
            ):  # we assume proper subspace through origin
                heuristic_mse_shrunken_on_train = loss_if_predict_linalg_shrunken(
                    loss_func,
                    train_loader,
                    "train",
                    sigma2_pure_context,
                    sigma2_corruption,
                    style_origin_subspace=style_origin_subspace,
                    style_corruption_orthog=style_corruption_orthog,
                )
                heuristic_mse_shrunken_on_test = loss_if_predict_linalg_shrunken(
                    loss_func,
                    test_loader,
                    "test",
                    sigma2_pure_context,
                    sigma2_corruption,
                    style_origin_subspace=style_origin_subspace,
                    style_corruption_orthog=style_corruption_orthog,
                )
                assert style_subspace_dimensions == "random"
                dim_d_k = np.random.randint(
                    1, min(dim_n, context_len // 2), size=num_W_in_dataset
                )
                theory_expected_error_linalg_shrunken = theory_linear_expected_error(
                    dim_n, dim_d_k, sigma2_corruption, linear_sigma2_pure_context
                )  # style_subspace_dimensions is a string "random" TODO@DR not sure if correct

                loss_vals_dict["baselines"]["loss_if_predict_linalg_shrunken"] = dict(
                    alias="heuristic_proj_shrunken",
                    label=r"$\gamma P \tilde x$",
                    val_train=heuristic_mse_shrunken_on_train,
                    val_test=heuristic_mse_shrunken_on_test,
                    pltkwargs=dict(color="mediumpurple"),
                )

                loss_vals_dict["baselines"]["theory_expected_error_linalg_shrunken"] = (
                    dict(
                        alias="theory_expected_error_linalg_shrunken",
                        label=r"$\mathbb{E}[L(\theta^*)]$",
                        val_train=theory_expected_error_linalg_shrunken,  # note train/test don't matter - theory curve
                        val_test=theory_expected_error_linalg_shrunken,
                        pltkwargs=dict(color="mediumpurple", linestyle=":"),
                    )
                )

    plt.plot(curve_y_losstrain_epochs_avg)
    plt.show()  # I can see epoch loss decreasing down to 0.7786882519721985 with 80 datapoints dim 32 context 500 linear 10 epoch

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

    if args.model["family"] in {"gpt2"}:
        model = build_model(args.model)
    else:
        # model = TransformerModelV1nores(args.training["context_len"], args.training["dim_n"]) #this seems to fit the best for the default
        model = TransformerModelV1noresOmitLast(
            args.training["context_len"], args.training["dim_n"]
        )
        # model = TransformerModelV3(
        #    args.training["context_len"], args.training["dim_n"] #very very different
        #  )

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

    if args.training["nn_model"] not in ["TransformerModelQKVnores"]:
        learned_W_KQ = net.W_KQ.detach().cpu().numpy()
        learned_W_PV = net.W_PV.detach().cpu().numpy()
        if (
            learned_W_KQ.size == 1
        ):  # in this case we are training 1-param weights (scaled identity) - remake as arr
            learned_W_KQ = learned_W_KQ * np.eye(args.training["dim_n"])
            learned_W_PV = learned_W_PV * np.eye(args.training["dim_n"])

        vis_weights_kq_pv(
            learned_W_KQ,
            learned_W_PV,
            titlemod=r"$\theta$ final",
            dir_out=io_dict["dir_vis"],
            fname="weights_final",
            flag_show=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for the denoising icl task, data gen, train, and eval."
    )
    parser.add_argument("--config-file", help="Path to YAML config file")

    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
            parser.set_defaults(**config)
        args = parser.parse_args()  # Reload arguments to apply YAML values

    assert args.model["family"] in [
        "gpt2",
        "1linearT1H",
        "1softmaxT1H",
        "2softmaxT1H",
        "2softmaxTres",
        "1softmaxT2H",
        "1linearT2H",
        "tanhT",
    ]
    print(f"Running with: {args}")

    with open(os.path.join(DIR_OUT, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)

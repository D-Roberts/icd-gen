import os
import sys
from pathlib import Path
import numpy as np
import comet_ml
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.datagen import (
    PreBatchedDataset,
    train_batched_data,
    test_batched_data,
)

from EnerdiT import *

API_KEY = Path(".comet_api").read_text().strip()

comet_ml.login()
workspace = Path(".comet_workspace").read_text().strip()

exp = comet_ml.Experiment(
    api_key=API_KEY,
    project_name="enerdit",
    workspace=workspace,
    auto_metric_logging=True,  # default
)

from transformers.optimization import get_cosine_schedule_with_warmup


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


class Trainer:
    """
    To train the EnerdiT.
    """

    def __init__(
        self,
    ):
        pass

    def train_step(
        self, model, xs, ys, optimizer, loss_funct, loss_funcs, t=1, lamu=0.001
    ):
        optimizer.zero_grad()

        energy, space_score, time_score = model(xs)

        ############################################Losses - TODO@DR - should
        ########factor out / separate into its own Loss class

        # # query energy
        # qenergy = energy[:, -1]

        # # print("in train step device of xs ys", xs.device, ys.device)

        # loss = 0

        # # this is th portion
        # if loss_funct:  # so calculating this on the query and its label
        #     loss1 = loss_funct(time_score, xs[:, :, -1], ys, t=t)
        #     loss += loss1

        # # this is sh portion
        # if loss_funcs:
        #     loss2 = loss_funcs(space_score, xs[:, :, -1], ys, t=t)
        #     loss += loss2

        # # Test an added direct U component to tighten the feedback loop loss-preds (an NLL afterall)
        # # Here adding the average over context but with a hyperparam

        # loss += lamu * energy.mean()
        ############################Here above is the new loss to debug and tune
        #######later, after archi ###############

        # for patch diffusion loss from EDM
        # print(f"shape of score {score.shape} and x {xs.shape}")
        preds = torch.permute(space_score, (0, 2, 1))
        loss = loss_funcs(preds, xs)

        loss.backward()
        optimizer.step()

        return (
            loss.detach().item(),
            energy.detach(),  # take out the energy for the context to analyze
            space_score.detach(),
            time_score.detach(),
        )

    def train(
        self,
    ):
        pass

    def eval(
        self,
    ):
        pass


# Ad hoc test

# trainer = Trainer()

# model = EnerdiT(
#     batch=2,
#     context_len=5,
#     d_model=32,
#     input_dim=8,
#     cf1_init_value=0.5,  # This is just to init - but param is learned.
#     num_heads=1,
#     depth=1,
#     mlp_ratio=4,
# )

# xs = torch.randn(2, 8, 5, requires_grad=True)
# ys = torch.randn(2, 1, requires_grad=True)


# page 4
class TimeLoss(nn.Module):
    def __init__(self, d_model, reduction="mean"):
        super(TimeLoss, self).__init__()
        self.d_model = d_model

    def forward(self, preds, x, y, t=1):
        """take average over minibatch in this implementation
        this is for one time t;

        TODO@DR: reason how to handle t in my
        case.

        X: (b, fused_patch_dim) this is the patch we are predicting on
        y: this is the target
        """

        # For right now get Eq 4 with only the predicting patch x and y
        dUt = 0.5 * ((self.d_model - torch.norm(y - x, p=2) ** 2)).mean()

        # Eq 5: TODO@DR: one mean or two means now?

        ltsm = ((preds - dUt) ** 2).mean()
        return ltsm


class SpaceLoss(nn.Module):
    def __init__(self):
        super(SpaceLoss, self).__init__()
        pass

    def forward(self, preds, x, y, t=1):
        """
        Equation 3; again not sure how will treat time step here
        since query is only one time step.

        """
        duy = (y - x).mean()

        # Again here preds should be grad of U(y,t) wrt to y
        # and this loss is for time step t; not sure in my case
        ldsm = ((torch.norm(preds - duy, p=2)) ** 2).mean()

        return ldsm


trainer = Trainer()


model = EnerdiT(
    batch=4,
    context_len=10,
    d_model=32,
    input_dim=200,
    cf1_init_value=0.5,  # This is just to init - but param is learned.
    num_heads=1,
    depth=1,
    mlp_ratio=4,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,
    betas=(
        0.9,
        0.98,
    ),
    eps=1e-8,
    weight_decay=0.0,
)

# For archi debug - use some loss, let's do MSE

loss_func_debug = nn.MSELoss()

loss_funct = TimeLoss(d_model=32)
loss_funcs = SpaceLoss()

# from patch diffusion; not used
# patchd_loss = PatchDiffLoss()
# xs = get_noised_forpatchdiff(xs)

# print(xs.shape)

# print(model)


# TODO@Note that I could augment the diffusion process with the
# context for some time steps and then let a few time steps
# in a sort of multitask learning fashion. Not sure, we'll see.

# TODO@ recall torch.bmm might be faster if I need a matmul

# TODO@DR explore the compute graph for shortcuts

# time schedule from DiT
# t = torch.randint(
#     0, 3, (4,)
# )  # gen ints in between 3 and 3 is number of time steps and 4 is batch size
# print("generated time schedule as in dit ", t)  #

###############################Now Train with some synthetic batches generated
###with one structure per batch

# Custom batches with the structure per batch already batched
train_set = PreBatchedDataset(train_batched_data)
test_set = PreBatchedDataset(test_batched_data)

train_size = len(train_batched_data)
train_loader = DataLoader(
    train_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
)

test_loader = DataLoader(
    test_set, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
)

###########################POC train on the synthetic batches as they are right now
epochs = 3
# batchsize is set in datagen for this synthetic batch group structure setup
lamu = 0.001  # this is hyperpar for U regularizer

print(model)

for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, target = data

        print(f"inputs {inputs.shape} and targets {target.shape}")
        # (4, 200, 10)

        # Pass None if only one component of the loss is used
        loss, energy, sh, th = trainer.train_step(
            model, inputs, target, optimizer, None, loss_func_debug, t=1, lamu=lamu
        )

        # What would t be for me?
        print(f"loss is {loss}\n")

        # print(f"energy {energy} and shape {energy.shape} \n")
        # # yes looks fine shapewise, one energy per batch per context token

        # print(f"space score shape {sh.shape} and values {sh}\n")
        # print(f"time score shape {th.shape} and values {th} \n")

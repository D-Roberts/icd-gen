import os
import sys
from pathlib import Path
import numpy as np
import comet_ml
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from groups_datagen import (
#     x_train,
#     y_train,
#     x_test,
#     y_test,
#     PreBatchedDataset,
#     train_batched_data,
#     test_batched_data,
# )

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
    To train the EnerDiT
    """

    def __init__(
        self,
    ):
        pass

    def train_step(self, model, xs, ys, optimizer, loss_funct, loss_funcs, t=1):
        optimizer.zero_grad()
        energy, space_score, time_score = model(xs)
        qenergy = energy[:, -1]

        # TODO@DR it isn't the energy that should go in loss but I want it for density and analysis

        # print("in train step device of xs ys", xs.device, ys.device)

        loss = 0
        # for TimeLoss func
        if loss_funct:  # so calculating this on the query and its label
            loss1 = loss_funct(time_score, xs[:, :, -1], ys, t=t)
            loss += loss1

        if loss_funcs:
            loss2 = loss_funcs(space_score, xs[:, :, -1], ys, t=t)
            loss += loss2

        # for patch diffusion loss from EDM
        # print(f"shape of score {score.shape} and x {xs.shape}")
        # preds = torch.permute(score, (0, 2,1))
        # loss = loss_funcs(preds, xs)

        loss.backward()
        optimizer.step()

        return (
            loss.detach().item(),
            qenergy.detach(),
            space_score.detach(),
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

trainer = Trainer()

model = EnerdiT(
    batch=2,
    context_len=5,
    d_model=32,
    input_dim=8,
    num_heads=1,
    depth=1,
    mlp_ratio=4,
)

xs = torch.randn(2, 8, 5, requires_grad=True)
ys = torch.randn(2, 1, requires_grad=True)  # mimic energy


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
        # Also preds are the gradient of energy wrt to t here
        # So not very sure how this translates to my in-context set up
        # now

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


def get_weight_for_patch_loss():
    P_mean = -1.2
    P_std = 1.2
    sigma_data = 0.5

    rnd_normal = torch.randn([x.shape[0], 1, 1])
    sigma = (rnd_normal * P_std + P_mean).exp()
    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)

    return weight


# Just looking into ideas
class PatchDiffLoss(nn.Module):
    """Score matching variation
    Based on EDM. Based on "Elucidating ..."
    """

    def __init__(self):
        super(PatchDiffLoss, self).__init__()
        pass

    def forward(
        self, preds, x, labels=None, augment=None, t=1
    ):  # in patch diffusion y is confusingly the clean image, n is noise and y is label
        """
        from PatchDiffusion - simplified
        In patch diff they have the option for these cool geometric augmentations

        t is not needed in this model the noise is given as seen with the sigmas
        """

        # print(f"what is shape of weight {weight.shape}")
        # print(f"what is shape of x {x.shape}")
        # print(f"what is shape of preds {preds.shape}")

        weight = get_weight_for_patch_loss()

        ploss = weight * ((preds - x) ** 2)  # preds will be made on y + n

        return ploss.mean()


def get_noised_forpatchdiff(x):
    """x are images
    the sigma is the same as in the patchdiff loss
    This noise gets added to clean and goes into the net.
    In patchdiff - there are some additional c conditionings
    and skip added to the Unet output there
    """

    P_mean = -1.2
    P_std = 1.2
    sigma_data = 0.5

    rnd_normal = torch.randn([x.shape[0], 1, 1])

    sigma = (rnd_normal * P_std + P_mean).exp()
    n = torch.rand_like(x) * sigma
    # print(f"shape of n {n.shape} and of sigma {sigma.shape}")

    return x + n


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

# loss_func = nn.MSELoss()
loss_funct = TimeLoss(d_model=32)
loss_funcs = SpaceLoss()

# from patch diffusion
patchd_loss = PatchDiffLoss()
xs = get_noised_forpatchdiff(xs)

print(xs.shape)


# Pass None if only one component of the loss is used
loss, output, score = trainer.train_step(
    model, xs, ys, optimizer, loss_funct, loss_funcs, t=1
)

# What would t be for me?
print(f"loss is {loss}\n")
print(f"output {output}\n")

print(f"intermed {score.shape}\n")


# TODO@Note that I could augment the diffusion process with the
# context for some time steps and then let a few time steps
# in a sort of multitask learning fashion.

# TODO@ recall torch.bmm might be faster if I need a matmul

# TODO@DR explore the compute graph for shortcuts

# time schedule from DiT
t = torch.randint(
    0, 3, (4,)
)  # gen ints in between 3 and 3 is number of time steps and 4 is batch size
print("generated time schedule as in dit ", t)  #

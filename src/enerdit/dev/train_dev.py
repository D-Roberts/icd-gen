"""
Dev archi layer by layer
"""

import os
import sys
from pathlib import Path
import numpy as np
import comet_ml
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datagen_onestr_dev import train_loader, test_loader

from EnerdiT_dev import *

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
    To dev the EnerdiT.
    """

    def __init__(
        self,
    ):
        pass

    def train_step(
        self,
        model,
        xs,
        ys,
        optimizer,
        loss_func,
        t=1,
    ):
        optimizer.zero_grad()

        energy, space_score, time_score = model(xs)

        # Use the ys label right now with l1 loss
        # For architecture dev

        preds = torch.permute(space_score, (0, 2, 1))
        # print(f"for L1 preds which are the space_score {preds.shape} and targets {ys.shape}")

        # FOr l1 loss calculation - use last score on the query token
        loss = loss_func(preds[:, :, -1], ys)

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


trainer = Trainer()


model = EnerdiT(
    context_len=8,
    d_model=4,
    input_dim=128,
    cf1_init_value=0.5,  # This is just to init - but param is learned.
    num_heads=1,
    depth=1,
    mlp_ratio=4,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(
        0.9,
        0.98,
    ),
    eps=1e-8,
    weight_decay=0.0,
)

loss_func_dev = nn.L1Loss()


##############Dev train on simple one structure small dataset
epochs = 100

print(model)
model.to(device)

for epoch in range(epochs):
    print(f"***********Epoch is {epoch}")
    for i, data in enumerate(train_loader, 0):
        inputs, target = data

        loss, energy, sh, th = trainer.train_step(
            model, inputs.to(device), target.to(device), optimizer, loss_func_dev, t=1
        )

        print(f"loss is {loss}\n")
        # print(f"space score shape {sh.shape} and values {sh}\n") #. values are changing

        # CHeck that the weights are updating
        # for name, param in model.named_parameters():
        #     print(name)
        #     # if name == "space_head.space_head.weight": # yes weights are changing
        #     print(f"param is {param} and name is {name} ")

        # if i == 2:
        #     break

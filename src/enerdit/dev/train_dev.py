"""
Dev archi layer by layer
"""

import os
import sys
from pathlib import Path
import numpy as np
import comet_ml
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup

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
        st_loss,
        t,
    ):
        optimizer.zero_grad()

        qenergy, space_score, time_score = model(xs, t)

        space_score = torch.permute(space_score, (0, 2, 1))

        # Used the ys label right now with l1 loss for architecture dev
        # print(f"for L1 preds which are the space_score {preds.shape} and targets {ys.shape}")
        # FOr l1 loss calculation - use last score on the query token
        loss = st_loss(space_score[:, :, -1], ys[:, :64])  # use only the non-zero

        # here in code the label y is the clean
        # loss, loss_sp, loss_t = st_loss(
        #     space_score, time_score, xs, ys, t, return_both=True
        # )

        # print(f"in train step print space loss {loss_sp}")
        # print(f"in train step print time loss {loss_t}")

        loss.backward()
        optimizer.step()

        return (
            # loss_sp.detach().item(),
            # loss_t.detach().item(),  # for debug
            loss.detach().item(),
            qenergy.detach()
            .cpu()
            .numpy(),  # take out the energy for the context to analyze
            space_score.detach(),
            time_score.detach(),
        )

    def train_loop(
        self,
    ):
        pass

    def eval_step(
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


class TimeLoss(nn.Module):
    def __init__(self):
        super(TimeLoss, self).__init__()
        pass

    def forward(self, preds, x, y, tt):
        """take average over minibatch in this implementation
        this is for one time t;

        same as in space loss up until z calculation
        """
        bs, d, seq_len = x.shape
        # d is the patch size; going with the patch rather than the double patch here
        d = d // 2
        # extract the non-zero part only and the last token which is query noisy
        query = x[:, :d, -1]
        clean = y[:, :d]
        # print(f"in t loss query shape {query.shape}")
        # print(f"in sp loss clean shape {clean.shape}")

        # TODO@DR: would be best to get the z directly

        # t for this seq query last token
        t = tt[-1]

        # the noise
        z = (query - clean) * (1 / math.sqrt(t))
        # print(f"z in time loss when no noise {z}") # yes 0

        # print(f"preds shape in time loss {preds.shape}") #(B, seq_len)
        time_score = preds[:, -1]  # this extracts time score for query
        # print(f"time score when zero noise {time_score}") # around constant on some number - I suppose maybe it makes sense
        # based on inits

        # Added an average pooling to time head prediction layer so that it is
        # scalar since U is scalar and so is t
        # For right now - take the mean of the time score instead of
        # predicting a scalar directly TODO@DR reconsider

        # print(f"time score shape {time_score.shape}") # (B,)

        term1 = (t / d) * time_score
        znormedsq = torch.norm(z, p=2, dim=-1) ** 2
        # print(f"z normed shape {znormedsq.shape}")
        term2 = 0.5 * (1 - znormedsq / d)

        ltime = (term1 - term2) ** 2

        # print(f"ltime before minibatch mean {ltime.shape}")  # (B,) ok
        # mean over minibatch
        return ltime.mean()


class SpaceLoss(nn.Module):
    def __init__(self):
        super(SpaceLoss, self).__init__()
        pass

    def forward(self, preds, x, y, tt):
        """
        use t from query last token. tt is the t from uniform on seq

        """
        # print(f"what is y shape - the clean in SpaceLoss {y.shape}")
        # print(f"what is x shape - the noisy in SpaceLoss {x.shape}")
        # print(f"what is preds from Space Head shape - in SpaceLoss {preds.shape}")
        # so here x is (B, dim, seq_len) while y=clean target on last noisy query
        # is (B, dim)

        # get z from clean query, noisy query and t (assume t is not 0, which
        # should not be in training)
        bs, d, seq_len = x.shape
        # d is the patch size; going with the patch rather than the double patch here
        d = d // 2
        # extract the non-zero part only and the last token which is query noisy
        query = x[:, :d, -1]
        clean = y[:, :d]
        # print(f"query shape {query.shape}")
        # print(f"clean shape {clean.shape}")

        # t for this seq query
        t = tt[-1]

        # the noise
        z = (query - clean) * (1 / torch.sqrt(t))

        # the space head pred score
        sp = preds[:, :, -1]
        # space loss term1 (as in eq 43); non neg and non-zero
        term1 = math.sqrt(t / d) * sp
        term2 = z / math.sqrt(d)
        subtract = term1 - term2
        # print(f"in space loss subtr shape {subtract.shape}") #3, 64
        lspace = (torch.norm(subtract, p=2, dim=1)) ** 2
        # take norm over the input dim
        # print(f"what is lspace {lspace.shape} and over minibatch {lspace.mean()}")
        return lspace.mean()


class STLoss(nn.Module):
    """with space and time heads losses together"""

    def __init__(self):
        super(STLoss, self).__init__()
        self.spacel = SpaceLoss()
        self.timel = TimeLoss()

    def forward(self, space_scores, time_scores, sequence, label, t, return_both=True):
        spl = self.spacel(space_scores, sequence, label, t)
        tl = self.timel(time_scores, sequence, label, t)
        stl = spl + tl

        if return_both:
            return stl, spl, tl
        else:
            return stl


loss_func_dev = nn.L1Loss()

# spaceloss_dev = SpaceLoss()
# timeloss_dev = TimeLoss()
stloss_dev = STLoss()


# add a test
def test_eval(model, test_loader, loss_func_dev):
    count = 0.0
    test_loss_total = 0.0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            samples, targets = data
            energy, space_score, time_score = model(samples.to(device))

            # Use the ys label right now with l1 loss
            # For architecture dev

            preds = torch.permute(space_score, (0, 2, 1))
            # print(f"for L1 preds which are the space_score {preds.shape} and targets {ys.shape}")

            # FOr l1 loss calculation - use last score on the query token
            test_loss = loss_func_dev(preds[:, :, -1], targets.to(device)).item()

        test_loss_total += test_loss
        count += 1
    return test_loss_total / count


##############Dev train on simple one structure small dataset
epochs = 1
train_size = len(train_loader)


# scheduler = get_cosine_schedule_with_warmup(optimizer, 10, epochs * train_size)

# print(model)
model.to(device)


def normalize(inputs, target):
    in_min, in_max = torch.min(inputs), torch.max(inputs)
    target_min, target_max = torch.min(target), torch.max(target)
    range = in_max - in_min
    return (inputs - in_min) / range, (target - target_min) / (target_max - target_min)


batch_count = 0
energies = []
for epoch in range(epochs):
    print("***********STLoss Dev**********No Noise currently")
    print(f"***********Epoch is {epoch}")
    epoch_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, target = data

        #  all val bet 0 and 1
        inputs, target = normalize(inputs, target)

        # print(f"check range inputs {torch.min(inputs)}, {torch.max(inputs)}")
        # print(f"check range targets {torch.min(target)}, {torch.max(target)}")

        # print(f"one batch inputs are {inputs[0]} of shape {inputs[0].shape}") #[128, 8] so (2Xpatch dim, seqlen) for shape as expected
        # print(f"one batch label is {target[0]} of shape {target[0].shape}") # as expected
        # print(f"non zero part of label {target[0][:64]}") #as expected
        # print(f"zero part of label {target[0][64:]}") #as expected
        # print(f"one batch inputs first token {inputs[0][:, 0]} of shape {inputs[0][:, 0].shape}") #all nonzero, shape is (2Xpatch dim) ok
        # print(f"one batch inputs last token {inputs[0][:, -1]} of shape {inputs[0][:, -1].shape}") #half zeros, shape is (2Xpatch dim) ok
        # print(f"bec I took off noise last token and label should be equal {inputs[0][:, -1] == target[0]}") # yes, as expected
        b, pdim, seq_len = inputs.shape

        # THis will be the t to generate noise for the seq and to use in loss and in time embed
        t = torch.exp(
            torch.empty(seq_len).uniform_(math.log(10 ** (-9)), math.log(10**3))
        )
        # print(f"t min max is {torch.round(torch.min(t), decimals=6)} and {torch.round(torch.max(t), decimals=2)}")
        # t looks ok; so here is the same t tensor for each instance of this minibatch
        # TODO@DR: note that right now the same t will land on the last token / query for each instance
        # in the mini batch and will be used in noisy and in loss calculation. Must see about this.
        # maybe randomize per instance.

        # loss_sp, loss_t, loss, energy, sh, th = trainer.train_step(
        #     model,
        #     inputs.to(device),
        #     target.to(device),
        #     optimizer,
        #     stloss_dev,
        #     t.to(
        #         device
        #     ),  # t will be time embedded and added to the patch and space embeddings
        # )
        # # TODO@DR: after seeing learning with this loss, experiment with the
        # # energy regularizer too but not until loss goes down as is

        # test after time embeds with l1 on space score
        loss, energy, sh, th = trainer.train_step(
            model,
            inputs.to(device),
            target.to(device),
            optimizer,
            loss_func_dev,
            t.to(
                device
            ),  # t will be time embedded and added to the patch and space embeddings
        )

        # print(f"energy on query {energy.shape}") # this is for the minibatch
        # so let's just log the first

        print(f"total loss is {loss}\n")
        epoch_loss += loss
        batch_count += 1

        # exp.log_metrics(
        #     {"one -energy=logp in train batch": -energy[0]}, step=batch_count
        # )
        energies.extend(-energy)

        # TODO@DR should not have values outside 0,1 for p
        # exp.log_metrics(
        #     {"one p=exp(-en) in train batch": np.exp(-energy[0])}, step=batch_count
        # )

        exp.log_metrics({"batch loss": loss}, step=batch_count)
        # exp.log_metrics({"batch loss space": loss_sp}, step=batch_count)
        # exp.log_metrics({"batch loss time": loss_t}, step=batch_count)

        # exp.log_metrics({"Dev train spacehead-only batch query -energy=logp": -energy[:,-1]}, step=batch_count)
        # exp.log_metrics({"Dev train spacehead-only batch p": torch.exp(-energy)}, step=batch_count)
        # print(f"space score shape {sh.shape} and values {sh}\n") #. values are changing

        # CHeck that the weights are updating
    #     for name, param in model.named_parameters():
    #         print(name)
    #         if param.grad is not None:
    #             # if name == "space_head.space_head.weight": # yes weights are changing
    #             print(f"param is {param} and name is {name} and its grad is {torch.round(param.grad.cpu(), decimals=4)}")

    # # # scheduler.step()  # step the learning rate; if not cosine then no scheduler
    # print("\tlast LR:", scheduler.get_last_lr())

    # eval once per epoch in dev
    # leave the l1 for now
    # test_lossl1_mean = test_eval(model, test_loader, loss_func_dev)
    # print(f"Test l1 mean same set each epoch ***********{test_lossl1_mean}*******")
    # print(f"Train epoch loss {epoch_loss/train_size}*******")

    exp.log_metrics({"Dev Epoch loss": epoch_loss / train_size}, step=epoch)
    # exp.log_metrics({"Dev test each epoch l1 mean loss": test_lossl1_mean}, step=epoch)

    # stop for debug
    if batch_count == 3:
        break


################so when I train with no noise data, the losses have these non-zero values
# in train step print space loss 0.04604632779955864
# in train step print time loss 0.24885523319244385
# total loss is 0.2949015498161316


# print(len(energies))
# print(energies)
# # Create the histogram
# plt.hist(energies, bins=30, color='skyblue', edgecolor='black')

# # Add labels and a title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of logp')
# # Display the plot
# plt.show()

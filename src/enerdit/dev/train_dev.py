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

# from datagen_onestr_dev import train_loader, test_loader
from dgen_for_gaussian import get_batch_samples, train_loader, test_loader

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

torch.manual_seed(0)

model = EnerdiT(
    context_len=1,  # no context in simple; will have to change everything
    d_model=32,
    input_dim=1024,  # the way datagen is setup now - comes in one flattened, including in simple
    cf1_init_value=0.5,
    num_heads=1,
    depth=1,
    mlp_ratio=4,
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(
        0.9,
        0.999,
    ),
    # eps=1e-8,
    weight_decay=0.0,
)


class TimeLossV1(nn.Module):
    def __init__(self):
        super(TimeLossV1, self).__init__()
        pass

    def forward(self, time_score, z, t):
        """take average over minibatch in this implementation
        this is for one time t;

        only on the query TODO@DR: There is a bug in here see V2 if I go back to
        this.
        """
        bs, d = z.shape

        term1 = (t / d) * time_score
        znormedsq = torch.norm(z, p=2, dim=-1) ** 2
        # print(f"z normed shape {znormedsq.shape}")
        term2 = 0.5 * (1 - znormedsq / d)
        ltime = (term1 - term2) ** 2
        # print(f"ltime before minibatch mean {ltime.shape}")  # (B,) ok
        # mean over minibatch
        return torch.mean(ltime)


class TimeLossV2(nn.Module):
    def __init__(self):
        super(TimeLossV2, self).__init__()
        pass

    def forward(self, time_score, z, clean, query, t):
        """take average over minibatch in this implementation
        this is for one time t;

        only on the query so use y and x directly for signal
        supervision
        label is clean on the noisy query last sequence token
        """
        bs, d = z.shape  # on last token only

        weight_factor = (t / d) ** 2

        print(f"\ntime l begins*********************")

        print(f"time l weight_factor {weight_factor}")
        normedsq = torch.norm(query - clean, p=2, dim=-1) ** 2

        print(f"time l normedsq {normedsq.mean()}")  # one large
        parans_term2 = d / (2 * t) - normedsq / (2 * (t**2))
        print(f"d / (2 * t) {d / (2 * t)}")
        print(f"(2 * (t**2)) {(2 * (t**2))}")
        print(f"normedsq / (2 * (t**2)) {normedsq / (2 * (t**2))}")

        print(f"parans_term2 {parans_term2.mean()}")

        parans_term = time_score - parans_term2

        print(f"time_score {time_score.mean()}")
        time_match = weight_factor * (parans_term**2)
        # there is a pattern/signal in this suppervision on the time score
        # so the net should be able to learn it.
        # it can learn mse(y,x) right now so why not this?

        print(f"time_loss {time_match.mean()}")
        print(f"time l ends*********************")

        return torch.mean(time_match)


class SpaceLossV1(nn.Module):
    def __init__(self):
        super(SpaceLossV1, self).__init__()
        pass

    def forward(self, sp, z, t):
        """
        use t from query last token. tt is the t from uniform on seq
        sp and z correspond to query token

        TODO@DR: there is a bug on t I think - see message in TimeLossV1
        if I end up coming back to V1 but I am not certain this is
        the right supervision for the heads.

        """
        # print(f"what is y shape - the clean in SpaceLoss {y.shape}")
        # print(f"what is x shape - the noisy in SpaceLoss {x.shape}")
        # print(f"what is preds from Space Head shape - in SpaceLoss {preds.shape}")
        # so here x is (B, dim, seq_len) while y=clean target on last noisy query
        # is (B, dim)

        # get z from clean query, noisy query and t (assume t is not 0, which
        # should not be in training)
        bs, d = z.shape
        # print(f"d is ....{d}") # patch size (8x8 for example = 64)

        # space loss term1 (as in eq 43); non neg and non-zero
        term1 = torch.sqrt(t / d) * sp
        term2 = z / torch.sqrt(torch.tensor(d))

        # print(f"what is torch.sqrt(t / d) in space loss {torch.sqrt(t / d)}")
        subtract = term1 - term2
        # print(f"in space loss subtr shape {subtract.shape}") #3, 64

        lspace = (torch.norm(subtract, p=2, dim=1)) ** 2

        # test to see if grads if mse like loss
        # lspace = (torch.norm(sp - term2, p=2, dim=1)) ** 2

        # take norm over the input dim
        # print(f"what is lspace {lspace.shape} and over minibatch {lspace.mean()}")
        return torch.mean(lspace)


class SpaceLossV2(nn.Module):
    def __init__(self):
        super(SpaceLossV2, self).__init__()
        pass

    def forward(self, space_score, z, clean, query, t):
        """
        use t from query last token.
        same for z and query and label

        """
        print(f"what is shape - the clean in SpaceLoss {clean.shape}")
        print(f"what is shape - the query in SpaceLoss {query.shape}")
        print(f"what is shape - the z in SpaceLoss {z.shape}")

        # print(f"what is preds from Space Head shape - in SpaceLoss {preds.shape}")
        # so here x is (B, dim, seq_len) while y=clean target on last noisy query
        # is (B, dim)

        print(f"what is t here {t}")
        # now they match

        # sanity checks / debugging
        # result = torch.einsum('bd,b->bd', z, torch.sqrt(t))
        # print(f"check diff bet q and clean and comparison to zsqrtt {torch.mean(query - clean)} vs {torch.mean(result)}")
        # ok, same

        b, d = query.shape

        # print(f"d is ....{d}") # patch size (8x8 for example = 64)

        weight_factor = t / d  # this is safe, d is never 0

        print(f"weight_factor in spl {weight_factor}")
        # There was a bug here

        # multiply along batch dim
        term2 = torch.einsum("bd,b->bd", (query - clean), 1 / t)  # t not zero
        subtract = space_score - term2
        print(f"check values of space score {torch.mean(space_score)}")

        print(f"check values of squery {torch.mean(clean)}")
        print(f"check values of squery {torch.mean(query)}")
        print(f"check values of (query - clean) / t {torch.mean(term2)}")

        print(f"checkvalue of subtract in spacelossv2 {subtract.mean()}")
        print(f"check shape of subtract in spacelossv2 {subtract.shape}")

        ldsm = (torch.norm(subtract, p=2, dim=1)).pow(2)

        print(f"checkvalue of ldsm in spacelossv2 {ldsm.mean()}")

        lspace = weight_factor * ldsm
        # print(f"what is lspace {lspace.shape} and over minibatch {lspace.mean()}")
        return torch.mean(lspace)


class SpaceTimeLoss(nn.Module):
    """with space and time heads losses together"""

    def __init__(self):
        super(SpaceTimeLoss, self).__init__()
        self.spacel = SpaceLossV2()
        self.timel = TimeLossV2()

    def forward(
        self,
        space_scores,
        time_scores,
        z,
        clean,
        query,
        t,
        U,
        add_U=False,
        return_both=True,
    ):
        # print(f"z shape in loss {z.shape}") #[B, patch_dim, seq_len]
        spl = self.spacel(space_scores, z, clean, query, t)
        tl = self.timel(time_scores, z, clean, query, t)
        # print(f"t now in spacetime loss after I changed the schedule {t}")

        stl = spl + tl

        lamu = 0.001
        if add_U:
            stl += lamu * U.mean()  # minibatch average
            print(f"U in spacetime loss {U.mean()}")

        if return_both:
            return stl, spl, tl
        else:
            return stl


# loss_func_dev = nn.L1Loss() # this was for gamma;
loss_func_dev = nn.MSELoss()  # MSE looks better with gaussians
spacetimeloss_dev = SpaceTimeLoss()


class Trainer:
    """
    To dev the EnerdiT.
    """

    def __init__(
        self,
    ):
        pass

    def train_step(self, model, xs, ys, z, optimizer, spacetime_loss, t, dev_loss):
        optimizer.zero_grad()

        qenergy, space_score, time_score = model(xs, t)

        # no need for simple img level
        # space_score = torch.permute(space_score, (0, 2, 1))

        # print(f"so t is {t}")

        # put through only the query token (-1) and respectives for in-context
        # in simple, no context, no fused

        loss, loss_sp, loss_t = spacetime_loss(
            space_score,  # this is already shape of patch (not double)
            time_score,  # this is a scalar
            z,  # noise corresponding to query non-padded portion
            ys,  # clean label, non-padding portion
            xs,  # query last token of sequence non-padded portion
            t,  # t is just 1 per batch instance
            qenergy,
            add_U=True,  # if to add the energy regularizer to loss
            return_both=True,
        )

        # This is the MSE for dev purposes when working on archi or datagen
        # and not on losses
        # loss = dev_loss(space_score[:, :, -1], ys[:, :64])
        # it is learning with the target y on the space score or time score or sum.

        #######################################

        loss.backward()

        # Apply gradient clipping by norm; just leave it in although not needed
        # max_norm = 1.0  # Define the maximum allowed norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # After clipping, inspect gradient norms
        # there is no need for clipping so far.
        # print("\nGradients and weights")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         # print(f"  {name}: {param.grad.norm().item():.4f}")
        #         print(f"  {name}: {param.grad}")
        #         print(f"  {name}: {param}")

        optimizer.step()

        return (
            loss_sp.detach().item(),
            loss_t.detach().item(),  # for debug
            loss.detach().item(),
            qenergy.detach()
            .cpu()
            .numpy(),  # take out the energy for the context to analyze
            space_score.detach(),
            time_score.detach(),
        )


trainer = Trainer()

##############Dev train on simple one structure small dataset
epochs = 1
train_size = len(train_loader)

# scheduler = get_cosine_schedule_with_warmup(optimizer, 10, epochs * train_size)

print(model)
model.to(device)
print(f"train num batches {train_size}")

batch_count = 0
energies = []
for epoch in range(epochs):
    print(f"***********Epoch is {epoch}")
    epoch_loss = 0.0
    energy_epoch = 0.0

    for i, data in enumerate(train_loader, 0):
        t, z, target, xs = get_batch_samples(data)
        print(f"batch index is {batch_count}")
        batch_count += 1
        print(f"t is {t}")
        print(f"\ntarget is {target}")
        print(f"\nxs is {xs}")

        # stop for debug
        if batch_count == 1:
            break

        # print(f"returned z.shape {z.shape}") #(B, patc, seq)
        # print(f"returned xs.shape {xs.shape}") #(B, 2patc, seq)
        # print(t) # a seq len
        # print(f"target shape {target.shape}") #(B, 2patch)

        # TODO@DR: note that right now the same t will land on the last token / query for each instance
        # in the mini batch and will be used in noisy and in loss calculation. Must see about this.
        # maybe randomize per instance.

    #     loss_sp, loss_t, loss, energy, sh, th = trainer.train_step(
    #         model,
    #         xs.to(device),
    #         target.to(device),
    #         z.to(device),
    #         optimizer,
    #         spacetimeloss_dev,
    #         t.to(
    #             device
    #         ),
    #         loss_func_dev,
    #     )

    #     # print(f"energy on query {energy.shape}") # this is for the minibatch
    #     # so let's just log the first

    #     # print(f"space loss is {loss_sp}\n")
    #     # print(f"time loss is {loss_t}\n")
    #     print(f"total loss is {loss}\n")

    #     epoch_loss += loss
    #     batch_count += 1
    #     energy_epoch += energy.sum()

    #     energies.extend(-energy)

    #     # TODO@DR should not have values outside 0,1 for p
    #     # exp.log_metrics(
    #     #     {"one p=exp(-en) in train batch": np.exp(-energy[0])}, step=batch_count
    #     # )

    #     exp.log_metrics({"batch loss": loss}, step=batch_count)
    #     exp.log_metrics({"batch loss space": loss_sp}, step=batch_count)
    #     exp.log_metrics({"batch loss time": loss_t}, step=batch_count)

    #     # CHeck that the weights are updating (look at several)
    #     for name, param in model.named_parameters():
    #         print(name)
    #         if param.grad is not None:
    #             # if name == "space_head.space_head.weight": # yes weights are changing
    #             print(f"param is {param} and name is {name} and its grad is {torch.round(param.grad.cpu(), decimals=4)}")

    # exp.log_metrics({"Dev Epoch loss": epoch_loss / train_size}, step=epoch)
    # exp.log_metrics(
    #     {"avg epoch energy aka nll": energy_epoch / batch_count * train_size},
    #     step=epoch,
    # )

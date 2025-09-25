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
from dgen_for_gaussian import (
    get_batch_samples,
    train_loader,
    test_loader,
    get_batch_samples_test,
    get_batch_samples_w_context,
)

from src.enerdit.dev.densities_dev import *

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

model = DensityTranformer(
    d_model=256,  # 512
    input_dim=1024,  # the way datagen is setup now - comes in one flattened, including in simple
    cf1_init_value=0.5,
    num_heads=1,
    depth=1,
    mlp_ratio=4,
    patch_dim=2,
    context_len=None,  # no context in simple; will have to change everything
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

    def forward(self, time_score, z, clean, query, t):
        """take average over minibatch in this implementation
        ; p19

        """
        bs, d = z.shape
        weight_factor = t / d
        # print(f"t shape in t loss v1 {t.shape}")
        # print(f"weight_factor th {weight_factor.shape}")
        # print(f"time_score {time_score.squeeze().shape}") # comes with 1 extra dim
        term1 = weight_factor * time_score.squeeze()

        znormedsq = torch.norm(z, p=2, dim=-1) ** 2
        # print(f"z normed shape {znormedsq.shape}")
        term2 = 0.5 * (1 - znormedsq / d)
        ltime = (term1 - term2) ** 2
        # print(f"ltime before minibatch mean {ltime.shape}")
        # mean over minibatch
        return torch.mean(ltime)


class TimeLossV2(nn.Module):
    def __init__(self):
        super(TimeLossV2, self).__init__()
        pass

    def forward(self, time_score, z, clean, query, t):
        """p4; prediction is scalar; t is sampled for thebatch"""
        bs, d = z.shape  # on last token only

        weight_factor = (t / d) ** 2

        # print(f"\ntime l begins*********************")

        # print(f"time l weight_factor {weight_factor}")
        normedsq = torch.norm(query - clean, p=2, dim=-1) ** 2

        # print(f"time l normedsq {normedsq.mean()}")  # one large
        parans_term2 = d / (2 * t) - normedsq / (2 * (t**2))
        # print(f"d / (2 * t) {d / (2 * t)}")
        # print(f"(2 * (t**2)) {(2 * (t**2))}")
        # print(f"normedsq / (2 * (t**2)) {normedsq / (2 * (t**2))}")

        # print(f"parans_term2 {parans_term2.mean()}")

        parans_term = time_score - parans_term2

        # print(f"time_score {time_score.mean()}")
        time_match = weight_factor * (parans_term**2)
        # there is a pattern/signal in this suppervision on the time score
        # so the net should be able to learn it.
        # it can learn mse(y,x) right now so why not this?

        # print(f"time_loss {time_match} and over mini batch {time_match}")
        # print(f"time l ends*********************")

        return torch.mean(time_match)


class SpaceLossV1(nn.Module):
    def __init__(self):
        super(SpaceLossV1, self).__init__()
        pass

    def forward(self, space_score, z, clean, query, t):
        """
        p19
        sp is space head prediction of shape of z

        """
        # print(f"what is y shape - the clean in SpaceLoss {y.shape}")
        # print(f"what is x shape - the noisy in SpaceLoss {x.shape}")
        # print(f"what is preds from Space Head shape - in SpaceLoss {preds.shape}")

        # get z from clean query, noisy query and t (assume t is not 0, which
        # should not be in training)
        bs, d = z.shape
        # print(f"d is ....{d}") #

        # space loss term1 (as in eq 43); non neg and non-zero
        weight_term1 = torch.sqrt(t / d)
        # print(f"in space loss weight_term1 shape {weight_term1.shape}")
        # print(f"in space loss space_score shape {space_score.shape}")

        # multiply along batch dim
        term1 = torch.einsum("bd,b->bd", space_score, weight_term1)
        term2 = z / torch.sqrt(torch.tensor(d))

        # print(f"in space loss term2 shape {term2.shape}")
        # print(f"what is torch.sqrt(t / d) in space loss {torch.sqrt(t / d)}")
        subtract = term1 - term2
        # print(f"in space loss subtr shape {subtract.shape}") #3, 64

        # dim 1 is the input dimension (not the batch or the context when context)
        lspace = (torch.norm(subtract, p=2, dim=1)) ** 2

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
        # print(f"what is shape - the clean in SpaceLoss {clean.shape}")
        # print(f"what is shape - the query in SpaceLoss {query.shape}")
        # print(f"what is shape - the z in SpaceLoss {z.shape}")

        # print(f"what is preds from Space Head shape - in SpaceLoss {preds.shape}")
        # so here x is (B, dim, seq_len) while y=clean target on last noisy query
        # is (B, dim)

        # print(f"what is t here {t}")
        # now they match

        # sanity checks / debugging
        # result = torch.einsum('bd,b->bd', z, torch.sqrt(t))
        # print(f"check diff bet q and clean and comparison to zsqrtt {torch.mean(query - clean)} vs {torch.mean(result)}")
        # ok, same

        b, d = query.shape

        # print(f"d is ....{d}") # patch size (8x8 for example = 64)

        weight_factor = t / d  # this is safe, d is never 0

        # print(f"weight_factor in spl {weight_factor}")
        # There was a bug here

        # multiply along batch dim
        term2 = torch.einsum("bd,b->bd", (query - clean), 1 / t)  # t not zero
        subtract = space_score - term2
        # print(f"check mean values of space score in sp loss {torch.mean(space_score)}")

        # print(f"check values of squery {torch.mean(clean)}")
        # print(f"check values of squery {torch.mean(query)}")
        # print(f"check values of (query - clean) / t {torch.mean(term2)}")

        # print(f"checkvalue of subtract in spacelossv2 {subtract.mean()}")
        # print(f"check shape of subtract in spacelossv2 {subtract.shape}")

        ldsm = (torch.norm(subtract, p=2, dim=1)).pow(2)

        # print(f"checkvalue of ldsm in spacelossv2 {ldsm.mean()}")

        lspace = weight_factor * ldsm
        # print(f"what is lspace {lspace.shape} and over minibatch {lspace.mean()}")
        return torch.mean(lspace)


class SpaceTimeLoss(nn.Module):
    """with space and time heads losses together"""

    def __init__(self):
        super(SpaceTimeLoss, self).__init__()

        # self.spacel = SpaceLossV2()
        # self.timel = TimeLossV2()

        self.spacel = SpaceLossV1()
        self.timel = TimeLossV1()

    def forward(
        self,
        space_scores,
        time_scores,
        z,
        clean,
        query,
        t,
        U,
        lam_space,
        return_both=True,
    ):
        # print(f"z shape in loss {z.shape}") #[B, patch_dim, seq_len]
        spl = self.spacel(space_scores, z, clean, query, t)
        tl = self.timel(time_scores, z, clean, query, t)
        # print(f"t now in spacetime loss after I changed the schedule {t}")

        stl = lam_space * spl + tl

        if return_both:
            return stl, spl, tl
        else:
            return stl


# loss_func_dev = nn.L1Loss() # this was for gamma;
loss_func_dev = nn.MSELoss()  # MSE looks better with gaussians
spacetimeloss_dev = SpaceTimeLoss()


class Trainer:
    """
    To dev the DensityTranformer.
    """

    def __init__(
        self,
    ):
        pass

    def train_step(
        self,
        model,
        noisy,
        clean,
        noisy_context,
        clean_context,
        z,
        optimizer,
        spacetime_loss,
        t,
        dev_loss,
        device,
    ):
        optimizer.zero_grad()

        qenergy, space_score, time_score = model(
            noisy, noisy_context, clean_context, t, device
        )

        # no need for simple img level
        # space_score = torch.permute(space_score, (0, 2, 1))

        # print(f"so t is {t}")

        # put through only the query token (-1) and respectives for in-context
        # in simple, no context, no fused

        loss, loss_sp, loss_t = spacetime_loss(
            space_score,  # this is already shape of patch (not double)
            time_score,  # this is a scalar
            z,  # noise corresponding to query non-padded portion
            clean,  # clean label, non-padding portion
            noisy,  # query last token of sequence non-padded portion
            t,  # t is just 1 per batch instance
            qenergy,
            lam_space=1,  # make this hyper in yaml
            return_both=True,
        )

        # This is the MSE for dev purposes when working on archi or datagen
        # loss = dev_loss(space_score, clean)

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


def eval_test(model, criterion, dataloader, device):
    # print(f"loss func is {loss_func_dev}")
    mse_total = 0
    model.to(device)
    # model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # for dev
            if i == 2:
                break
            # t, z, clean, noisy = get_batch_samples_test(data)

            # add in context

            #     print(f"t is {t} of shape {t.shape}") # shape B
            #     print(f"\n clean is {clean} of shape {clean.shape}") #shape (B, d)
            #     print(f"\n noisy is {noisy} of shape {noisy.shape}")
            #     print(f"\nz is {z} of shape {z.shape}")
            #     sqrttz = torch.einsum("bd,b->bd", z, torch.sqrt(t))
            #     print(
            #     f"compare noisy-clean with sqrt(t)z {torch.mean(noisy-clean-sqrttz)} elemwise {(noisy-clean)[0]} vs {sqrttz[0]}"
            # )

            qenergy, space_score, time_score = model(noisy.to(device), t.to(device))
            loss = criterion(space_score, clean.to(device))  # l2.MSE per batch

            mse_total += loss

    mse = mse_total / len(dataloader)
    # psnr in db
    psnr = -10 * math.log10(mse)
    # print(f"for noise level t {t} psnr from mse over test set {psnr}")
    return psnr


##############Dev train on simple one structure small dataset
epochs = 1  # do a total of around 60; takes around 4hrs
# on mps with DensityTranformer 1 block 1 head with d_mod = 512
train_size = len(train_loader)

scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs * train_size)

print(model)
model.train()
model.to(device)
print(f"train num batches {train_size}")

batch_count = 0
energies = []
for epoch in range(epochs):
    epoch_loss = 0.0
    energy_epoch = 0.0

    for i, data in enumerate(train_loader):
        # t, z, clean, noisy = get_batch_samples(data)
        t, z, clean, noisy, noisy_context, clean_context = get_batch_samples_w_context(
            data
        )
        # print(f"batch index is {batch_count}")
        # stop for debug ********************************************
        # if batch_count == 3:
        #     break

        batch_count += 1
        # print(f"t is {t} of shape {t.shape}") # shape B
        # print(f"\n clean is {clean} of shape {clean.shape}") #shape (B, d)
        # print(f"\n noisy is {noisy} of shape {noisy.shape}")
        # print(f"\n clean_context is {clean_context} of shape {clean_context.shape}") #shape (B, d)
        # print(f"\n noisy_context is {noisy_context} of shape {noisy_context.shape}")
        # print(f"\nz is {z} of shape {z.shape}")
        # sqrttz = torch.einsum("bd,b->bd", z, torch.sqrt(t))
        # print(
        #     f"compare noisy-clean with sqrt(t)z {torch.mean(noisy-clean-sqrttz)} elemwise {(noisy-clean)[0]} vs {sqrttz[0]}"
        # )

        # they do seem equal
        # data from simple task looks ok so far; noisy = clean +sqrt(t)z
        # with clean from N(0, Id)*4 (stdev = 4) and z from N(0, I)
        # I put t in 0.01 to 100
        # b = 20, d = 1024 here; 4000 train batches
        # or b = 512; train = 156

        # TODO@DR: note that right now the same t will land on the last token / query for each instance
        # in the mini batch and will be used in noisy and in loss calculation. Must see about this.
        # maybe randomize per instance. [for in-context patch level problem]

        loss_sp, loss_t, loss, energy, sh, th = trainer.train_step(
            model,
            noisy.to(device),
            clean.to(device),
            noisy_context.to(device),
            clean_context.to(device),
            z.to(device),
            optimizer,
            spacetimeloss_dev,
            t.to(device),
            loss_func_dev,
            device,
        )

        #     # print(f"energy on query {energy.shape}") # this is for the minibatch
        #     # so let's just log the first

        #     # print(f"space loss is {loss_sp}\n")
        #     # print(f"time loss is {loss_t}\n")
        print(f"total loss is {loss}\n")
        print(f"space loss is {loss_sp}\n")
        print(f"time loss is {loss_t}\n")
        print(f"U batch first {energy[0]}\n")
        # print(f"t {t}\n")

        epoch_loss += loss

        #     energies.extend(-energy)

        exp.log_metrics({"batch loss": loss}, step=batch_count)
        exp.log_metrics({"batch loss space": loss_sp}, step=batch_count)
        exp.log_metrics({"batch loss time": loss_t}, step=batch_count)

        qnorm = torch.norm(noisy, p=2, dim=-1).mean()
        clean_norm = torch.norm(clean, p=2, dim=-1).mean()

        exp.log_metrics({"mean batch noisy norm": qnorm}, step=batch_count)
        exp.log_metrics({"mean batch clean norm": clean_norm}, step=batch_count)

        exp.log_metrics({"mean batch space pred": sh.mean()}, step=batch_count)
        exp.log_metrics({"batch loss space": loss_sp}, step=batch_count)
        exp.log_metrics({"batch loss time": loss_t}, step=batch_count)

        # Disable test to add in context
        # if i % 4 == 0:
        #     # higher is better
        #     psnr = eval_test(model, loss_func_dev, test_loader, device)
        #     exp.log_metrics({"psnr on test few batches t=15": psnr}, step=batch_count)

        exp.log_metrics(
            {"U mean batch": energy.mean()},
            step=batch_count,
        )
        # exp.log_metrics(
        #     {"exp(-U) first in batch": math.exp(-energy[0])},
        #     step=batch_count,
        # )

    #     # CHeck that the weights are updating (look at several)
    #     for name, param in model.named_parameters():
    #         print(name)
    #         if param.grad is not None:
    #             # if name == "space_head.space_head.weight": # yes weights are changing
    #             print(f"param is {param} and name is {name} and its grad is {torch.round(param.grad.cpu(), decimals=4)}")

    exp.log_metrics({"Dev Epoch loss": epoch_loss / train_size}, step=epoch)
    scheduler.step()  # step the learning rate; if not cosine then no scheduler
    print("\tlast LR:", scheduler.get_last_lr())

    print("end epoch:", epoch, "====================")


# comment out to add in context
# eval_test(model, loss_func_dev, test_loader, device)

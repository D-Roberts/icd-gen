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
    context_len=8,
    d_model=32,
    input_dim=128,  # the way datagen is setup now - comes in one flattened
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

        only on the query
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

        # time l begins*********************
        # time l weight_factor 1.896801116174629e-08
        # time l normedsq 9208.251953125
        # parans_term2 -59256904.0
        # time_score -0.0014224419137462974
        # time_loss 332987168.0
        # time l ends*********************

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
        # a very small number

        # mult a 0.5 from the analytical of U on term1
        subtract = term1 - term2
        # print(f"in space loss subtr shape {subtract.shape}") #3, 64

        # zero grads come due a lot to very small term1
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

        # subtract gets very high and then ldsm very very high
        # weight_factor in spl 1.942039951075003e-08
        # check values of space score 0.02238388918340206
        # check values of squery 0.5388038754463196
        # check values of squery 0.5314796566963196
        # check values of (query - clean) / t -5892.83837890625
        # checkvalue of subtract in spacelossv2 5892.86083984375
        # check shape of subtract in spacelossv2 torch.Size([5, 64])
        # checkvalue of ldsm in spacelossv2 1196198068224.0

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


# add a test
# def test_eval(model, test_loader, loss_func_dev, t):
#     count = 0.0
#     test_loss_total = 0.0

#     with torch.no_grad(): #TODO@DR this is not working now after refactor for gaussian
#         for i, data in enumerate(test_loader, 0):
#             t, z, samples, targets = get_batch_samples(data)
#             energy, space_score, time_score = model(samples.to(device), t)

#             # Use the ys label right now with l1 loss
#             # For architecture dev

#             preds = torch.permute(space_score, (0, 2, 1))
#             # print(f"for L1 preds which are the space_score {preds.shape} and targets {ys.shape}")

#             # FOr l1 loss calculation - use last score on the query token
#             test_loss = loss_func_dev(preds[:, :, -1], targets.to(device)).item()

#         test_loss_total += test_loss
#         count += 1
#     return test_loss_total / count


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

        space_score = torch.permute(space_score, (0, 2, 1))

        # FOr l1 or l2 loss calculation - use last score on the query token
        # loss_mse_spaces = st_loss(
        #     space_score[:, :, -1], ys[:, :64]
        # )  # use only the non-zero
        # # and on time score
        # loss_mse_timesc = st_loss(time_score[:, -1], ys[:, :64].mean(dim=-1))
        # loss = loss_mse_spaces + loss_mse_timesc

        # here in code the label y is the clean
        # getting losses on last token, the query

        # print(f"so t is {t}")

        # put through only the query token and respectives
        loss, loss_sp, loss_t = spacetime_loss(
            space_score[:, :, -1],  # this is already shape of patch (not double)
            time_score[:, -1],  # this is a scalar
            z[:, :64, -1],  # noise corresponding to query non-padded portion
            ys[:, :64],  # clean label, non-padding portion
            xs[:, :64, -1],  # query last token of sequence non-padded portion
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

        # print("Gradient norms before clipping- they are small not large:")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"  {name}: {param.grad.norm().item():.4f}")

        # Apply gradient clipping by norm; just leave it in although not needed
        # max_norm = 1.0  # Define the maximum allowed norm
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # After clipping, inspect gradient norms
        # there is no need for clipping but I'll just leave it for now.
        print("\nGradients and weights")
        for name, param in model.named_parameters():
            if param.grad is not None:
                # print(f"  {name}: {param.grad.norm().item():.4f}")
                print(f"  {name}: {param.grad}")
                print(f"  {name}: {param}")

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

    def train_loop(
        self,
    ):
        pass

    def eval_step(
        self,
    ):
        pass


trainer = Trainer()

##############Dev train on simple one structure small dataset
epochs = 60
train_size = len(train_loader)

# scheduler = get_cosine_schedule_with_warmup(optimizer, 10, epochs * train_size)

print(model)
model.to(device)

batch_count = 0
energies = []
for epoch in range(epochs):
    print(f"***********Epoch is {epoch}")
    epoch_loss = 0.0
    energy_epoch = 0.0

    for i, data in enumerate(train_loader, 0):
        t, z, target, xs = get_batch_samples(data)

        # print(f"returned z.shape {z.shape}") #(B, patc, seq)
        # print(f"returned xs.shape {xs.shape}") #(B, 2patc, seq)
        # print(t) # a seq len
        # print(f"target shape {target.shape}") #(B, 2patch)

        # TODO@DR: note that right now the same t will land on the last token / query for each instance
        # in the mini batch and will be used in noisy and in loss calculation. Must see about this.
        # maybe randomize per instance.

        loss_sp, loss_t, loss, energy, sh, th = trainer.train_step(
            model,
            xs.to(device),
            target.to(device),
            z.to(device),
            optimizer,
            spacetimeloss_dev,
            t.to(
                device
            ),  # t will be time embedded and added to the patch and space embeddings
            loss_func_dev,
        )
        # # TODO@DR: after seeing learning with this loss, experiment with the
        # # energy regularizer too but not until loss goes down as is

        # test after time embeds with l1 on space score
        # loss, energy, sh, th = trainer.train_step(
        #     model,
        #     xs.to(device),
        #     target.to(device),
        #     optimizer,
        #     loss_func_dev,
        #     t.to(
        #         device
        #     ),  # t will be time embedded and added to the patch and space embeddings
        # )

        # print(f"energy on query {energy.shape}") # this is for the minibatch
        # so let's just log the first

        # print(f"space loss is {loss_sp}\n")
        # print(f"time loss is {loss_t}\n")
        print(f"total loss is {loss}\n")

        epoch_loss += loss
        batch_count += 1
        energy_epoch += energy.sum()

        energies.extend(-energy)

        # TODO@DR should not have values outside 0,1 for p
        # exp.log_metrics(
        #     {"one p=exp(-en) in train batch": np.exp(-energy[0])}, step=batch_count
        # )

        exp.log_metrics({"batch loss": loss}, step=batch_count)
        exp.log_metrics({"batch loss space": loss_sp}, step=batch_count)
        exp.log_metrics({"batch loss time": loss_t}, step=batch_count)

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
    # with MSE in gaussians
    # test_lossmse_mean = test_eval(model, test_loader, loss_func_dev, t)
    # print(f"Test l1 mean same set each epoch ***********{test_lossmse_mean}*******")

    exp.log_metrics({"Dev Epoch loss": epoch_loss / train_size}, step=epoch)
    exp.log_metrics(
        {"avg epoch energy aka nll": energy_epoch / batch_count * train_size},
        step=epoch,
    )
    # exp.log_metrics({"Dev test each epoch l1 mean loss": test_lossl1_mean}, step=epoch)

    # stop for debug
    # if batch_count == 2:
    #     break

###########################later
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

# in gamma related problems negative log-likelihood is typically employed
# so consider using a KL based or Jensen-Shannon divergence based criterion

import torch
import torch.nn.functional as F
from vit_spatial_poc import net
import torch.nn as nn


def JSD(p, q, reduction: str = "batchmean") -> torch.Tensor:
    """
    Jensen-Shannon divergence is symmetrized and smoothed version of KL.

    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    p and q must be tensors of log-probabilities.

    DR: I surmise appropriate for energy based because I need probabilities.
    TBD@DR: how and if to use. If I end up using a KL - use JSD instead.

    """
    # calculate mixture distribution M
    m = 0.5 * (p.exp() + q.exp()).log()

    # calculate KL for each term
    kl_pm = F.kl_div(p, m, reduction=reduction, log_target=True)
    kl_qm = F.kl_div(q, m, reduction=reduction, log_target=True)

    # GET jsd
    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd


# test JSD - TODO@DR put log_softmax in and use it.
log_probs_p = F.log_softmax(torch.randn(10, 5), dim=-1)
log_probs_q = F.log_softmax(torch.randn(10, 5), dim=-1)
print(log_probs_p.shape)
jsd_value_batchmean = JSD(log_probs_p, log_probs_q, reduction="batchmean")
print("calculate JSD ", jsd_value_batchmean)


# ------------------------
# Calculate SURE (Stein Unbiased ) - following paper ``Generalizat diffusion
# harmonic ...iclr24, Kad, G, etc''

# though the code seems to just use the MSE

# Note that the MSE uses sigma corresponding to noise level
# For Gamma, noise is characterized by the 2 parameters.
# Also likely MSE loss is hardly appropriate but for now use this
# untheoretically unsound heuristic

# estimate gamma noise level by its variance for one group
galpha = 1
gbeta = 0.8
gamma_noise = galpha * gbeta * gbeta
print(gamma_noise)

# @DR see how the jacob is done here but if I use - use the efficient jacrev
# from https://github.com/LabForComputationalVision/memorization_generalization_in_diffusion_models/blob/main/code/linear_approx.py
# def calc_jacobian( inp, model,layer_num=None, channel_num=None):


# Alternative for jacobian - try use the more performant jacrev
# https://docs.pytorch.org/functorch/stable/generated/functorch.jacrev.html
def create_an_f(weight, bias, x):
    return torch.nn.functional.linear(x, weight, bias)


#  AD hoc test jacob Example inputs
weight = torch.randn(2, 3)
bias = torch.randn(2)
x = torch.randn(3)

from torch.func import jacrev

# Compute jacob
jcb = jacrev(create_an_f, argnums=2)(weight, bias, x)
print("toy jcb", jcb)
print("jacobian trace ", torch.trace(jcb))
print("waht was funciton going in jacrev ", create_an_f)


class SURELoss(nn.Module):
    def __init__(self, noise_var, d):
        super(SURELoss, self).__init__()
        self.d = d
        self.noise_var = noise_var

    def forward(self, preds, label, jacob_trace):
        term1 = torch.norm(label - preds) ** 2

        term3 = -self.noise_var * self.d

        term2 = 2 * self.noise_var * jacob_trace

        return term1 + term2 + term3  # note this is sum not mean


# Ad hoc test SURELoss
sureloss = SURELoss(1, 2)
x = torch.randn(1, 4, 16)  # from the spatial
preds = net(x)  # this model isn't ready yet - still predicts a classif
print(preds.shape)
label = torch.randn(1)
jacob = jacrev(net, argnums=0)(x)  # recall that right now the net is still set up
# . for the classification problem
print("jacob is ", jacob[0][0][0][:].view((8, 8)))  # this should be square
jacob_trace = torch.trace(jacob[0][0][0][:].view((8, 8)))

sloss = sureloss(preds, label, jacob_trace)
print(f"poc SURE loss is {sloss}")


# TODO@DR: double check that jacob calculates what is necessary if I use this
# Also - really mse is not appropriate for gamma noise; neither is SURE
# because mixture of normal + gamma are another distrib; unless reason that
# patchwise sums - CLT

# TODO@DR - now with Gamma consider MAE, Jensen-SH, or GMMAD (median based) losses


##########################This is from patch diffusion / EDM ####
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

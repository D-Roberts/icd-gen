"""
simple one structure for enerdit dev
"""

import torch
from torch.distributions import Gamma
import torch.nn as nn
import math
import numpy as np

import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise


class DataSampler:
    def __init__(self):
        pass

    def sample_xs(self):
        raise NotImplementedError


class GroupSampler(DataSampler):
    """Generate a dataset with partitions of patche groups/sets.
    Each example does not have the same number of group members.
    Args:
        D - int - number of patches.
        sigma = level of gaussian noise controlling signal to noise specific to
        groups
        y - is a 1 or -1 label from the original classif task in jelassi paper
        keeping y around and tying gamma params to 1 or -1 [TODO@DR:
        still thinking how to structure this to be in-context but also learnable
        and not to mess up the group structures]

        N - num of samples (a sample should correspond to a patched up image
        but groups are constructed accross samples in the dataset generated)

        S = partitiion of indeces of patches in each group
        L = how many groups
        C = group cardinality

    """

    def __init__(
        self,
        N=20,
        D=8,
        d=None,
        L=None,
        S=None,
        C=None,
        sigma=None,
        q=None,
        gamma_dict=None,
    ):
        super().__init__()
        # per jelassi paper TODO@DR - make this more flexible
        self.N = N
        self.D = D
        if d is None:
            self.d = int(D**2)
        else:
            self.d = d

        if C is None:
            self.C = math.log(self.d) / 2
        else:
            self.C = C

        if q is None:
            self.q = math.log(self.d) / self.D
        else:
            self.q = q

        if L is None:
            self.L = int(self.D // self.C) + 1
        else:
            self.L = L

        if S is None:
            self.S = self.partition(self.D, self.L)
        else:
            self.S = S

        # not jelassi paper
        if gamma_dict is None:
            self.gamma_dict = {}  # create set gamma params;
            self.gamma_dict["y1"] = {
                "ga": torch.tensor([9.0]),
                "gb": torch.tensor([0.50]),
            }
            self.gamma_dict["y2"] = {"ga": torch.tensor([0.5]), "gb": torch.tensor([1])}
        else:
            self.gamma_dict = gamma_dict

    def partition(self, D, L, seed=42):
        """
        Get the
        TODO@DR: if I want every S set different, I must change the seed here.
        in-context variation task to task
        - this might be working ok already but double check.
        """
        # torch.manual_seed(seed)

        return torch.randperm(D).view(L, -1)

    def sample_xs(self, seeds=None):
        """w is the features we construct signal to noise for structure

        y is  from the classification task in jelassi paper.

        Control gamma noise param as a function of y.
        Instead could depend on the group index.

        On enerdit - put gaussian noise in instead TODO@DR: reason how
        this needs to change.

        Not sure yet how to best make in context, various options.
        Batch context with each batch a str seems apt.
        """

        # make a feature here; the signal in the dataset
        w = torch.randn(self.d)
        w /= w.square().sum().sqrt()

        y = torch.randn(
            self.N
        ).sign()  # this is what was the classification label in jelassi, 1 or -1, per instance
        # leave it
        q = math.log(self.d) / self.D
        sigma = 1 / math.sqrt(self.d)
        noise = (
            torch.randn(self.N, self.D, self.d) * sigma
        )  # gaussian noise with this sigma
        X = torch.zeros(self.N, self.D, self.d)

        for i in range(self.N):  # every example
            l = torch.randint(self.L, (1,))[0]  # a group index
            # print("l ", l)
            # print("partition", self.S)  #
            R = self.S[l]
            for j in range(self.D):
                if (
                    j in R
                ):  # this here creates artif groups by snr; X[i][j] is meant to be the pixel
                    X[i][j] = y[i] * w + noise[i][j]
                else:  # TODO@DR: rethink this SNR
                    prob = 2 * (torch.rand(1) - 0.5)
                    if prob > 0 and prob < q / 2:
                        delta = 1
                    elif prob < 0 and prob > -q / 2:
                        delta = -1
                    else:
                        delta = 0
                    X[i][j] = delta * w + noise[i][j]

        # TODO@DR: I might need to rethink how the patches in one sample are
        # and how in context is defined

        # last dim can be viewed as a patch flattened
        return X, y, w, self.S

    def add_gamma_noise(self, X_clean=None, y=None):
        if X_clean is None:  # should come in pair with y
            X_clean, y, w, _ = self.sample_xs()
        X_gnoisy = torch.zeros_like(X_clean)

        # Create two Gamma distribution objects with params tied to
        # y=1 or -1
        gamma_dist1 = Gamma(
            concentration=self.gamma_dict["y1"]["ga"], rate=self.gamma_dict["y1"]["gb"]
        )
        gamma_dist2 = Gamma(
            concentration=self.gamma_dict["y2"]["ga"], rate=self.gamma_dict["y2"]["gb"]
        )

        # TODO@DR I'll have to reason how the seeds for generators go here
        for i in range(X_clean.shape[0]):
            if y[i] == 1:
                gnoise = gamma_dist1.sample((1,))
            else:  # -1
                gnoise = gamma_dist2.sample((1,))

            # gamma is multiplicative
            noisy_patches = X_clean[i, :, :] * gnoise.view(-1, 1)
            X_gnoisy[i, :, :] += noisy_patches

        # so each instance will have noise from either gamma1 or gamma2 randomly as y is drawn but related to
        # how the underlying data is generated. Not sure if I want to do this or not.
        return X_clean, X_gnoisy, y

    def get_fused_sequence(self, X_clean=None, X_dirty=None):
        # just assume if X_clean given we also have X_dirty

        # TODO@dr I am calling it dirty bc I wonder about other distortions
        # beside noise; for instance those eigendistortions. Can
        # I do anything with that? Probably don't have time now.
        # Also those geometric transformations used for data augmentation
        # to make image models work better with smaller dataset (eg. in
        # patch diffusion). Can I repurpose them to create distortions?
        # Probably but don't have time now.

        if X_clean is None:
            X_clean, X_dirty, y = self.add_gamma_noise()

        # because the last gnoised patch should be the query and
        # its clean version the label, simply set the last clean
        # patch in the fused to 0

        # TODO@DR reason through how this padding affects the loss
        # calculation and the grad and if I need to do sth about it

        patch_dim = X_clean.shape[-1]
        label = torch.zeros((X_clean.shape[0], X_clean.shape[-1] * 2))
        label[:, :patch_dim] += X_clean[:, -1]

        X_clean[
            :, -1
        ] = 0.0  # 0 out last patch where the query is on the clean supervision

        # want to have dirty first in fused
        fused_seq = torch.cat((X_dirty, X_clean), dim=-1)

        # print(f"are dirty and clean diff? {X_dirty==X_clean}") #yes they are different

        return fused_seq, label


# Ad hoc testing
dggen = GroupSampler()
dataset, y, w, partition = dggen.sample_xs()
print(dataset.shape, y.shape)


_, noisy_d, _ = dggen.add_gamma_noise(dataset, y)
print(f"the noisy set {noisy_d.shape}")  # (num_samples, num_patches, flattened patch)
# In this setup num_patches (D) is tied to patch size (DxD=d)

fused_seq, label = dggen.get_fused_sequence(dataset, noisy_d)

# print(f"label shape {label.shape} and val {label[0]}") # looks ok
# print(f"check that last fused patch in teh seq has val 0 {fused_seq[0][-1].shape} and not equal to label {fused_seq[0][-1]} at the same pos") #yeap
# print(f"fused seq shape {fused_seq.shape}") #(B=20, seq_len=8, patch_dim=2*64=128)

print(f"feature w shape {w.shape}")  # 64 dim vector from d

print(f"partition of indices S {partition}")  # there are 10 groups with 2 elem each
# we have 8 patches; the partition is of 4 groups of 2 indeces each
# weach index is in each group changes but in this datagen the group cardinality
# is always 2
#  tensor([[0, 1],
# [3, 4],
# [7, 2],
# [5, 6]])


def plot_partition(D, partition):
    plt.figure(figsize=(4, 4))
    S_matrix = torch.eye(D, D)
    for x in partition:
        # print(f"wjat is x in S {x[1]}")
        S_matrix[x[0], x[1]] = 1 / 2
        S_matrix[x[1], x[0]] = 1 / 2
    plt.matshow(S_matrix, cmap="Greys")
    plt.axis("off")
    plt.savefig(f"src/enerdit/dev/one_str_dev.png")


plot_partition(8, partition)

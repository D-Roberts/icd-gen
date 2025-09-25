import torch
from torch.distributions import Gamma
import torch.nn as nn
import math
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Uniform
from torch.distributions import MultivariateNormal


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
        D=4,
        d=256,  # 16x16
        L=2,
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

    def partition(self, D, L, seed=42):
        """
        Get the
        TODO@DR: if I want every S set different, I must change the seed here.
        in-context variation task to task
        - this might be working ok already but double check.
        """
        # torch.manual_seed(seed)

        return torch.randperm(D).view(L, -1)

    def sample_patches(self, seeds=None):
        """w is the features we construct signal to noise for structure

        y is  from the classification task in jelassi paper.

        no noise here, this is for enerdit with gaussian.

        """

        # make a feature here; the signal in the dataset
        w = torch.randn(self.d)
        w /= w.square().sum().sqrt()

        y = torch.randn(
            self.N
        ).sign()  # this is what was the classification label in jelassi, 1 or -1, per instance
        # leave it
        # q = math.log(self.d) / self.D
        sigma = 1 / math.sqrt(self.d)
        noise = (
            torch.randn(self.N, self.D, self.d) * sigma
        )  # gaussian noise with this sigma
        X = torch.zeros(self.N, self.D, self.d)

        print(f"self q is {self.q}")

        num_times = 0

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
                    if prob > 0 and prob < self.q / 2:
                        delta = 1
                    elif prob < 0 and prob > -self.q / 2:
                        delta = -1
                    else:
                        delta = 0
                        num_times += 1
                    X[i][j] = delta * w + noise[i][j]

        print(f"how many times do I get noise only {num_times}")
        # last dim of X can be viewed as the cut up image put back together
        # and flattened
        return X.view(X.shape[0], -1), y, w, self.S


# Ad hoc testing
dggen = GroupSampler()

dataset, y, w, partition = dggen.sample_patches()
print(dataset.shape, y.shape)
# torch.Size([20, 1024])


print(partition)  # this mimics a general structure with 2 groups
# tensor([[3, 0],
#         [1, 2]])

# try it like this for now

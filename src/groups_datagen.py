"""
Patch generation module for new datagen strategies.

@DR: note that there is a torch.distributions.exp_family.ExponentialFamily 
abstract class to possibly work with in a more flexible/general datagen way
beyond normal.

"""
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.distributions import Gamma
import torch.nn as nn
import math


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
        sigma = level of gaussian noise when generated patch data,
        controls group structure

        N - num of samples (a sample would correspond to a patched up image)


    Returns:
        all dataset samples - torch.Tensor -
        of shape (num_samples, im_dim*im_dim)

    """

    def __init__(
        self,
        N=1000,
        D=10,
        d=None,
        L=None,
        S=None,
        C=None,
        sigma=None,
        q=None,
        gamma_dict=None,
    ):
        super().__init__()
        # per jelassi paper
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

        if gamma_dict is None:
            self.gamma_dict = {}  # create set gamma params
            self.gamma_dict["y1"] = {"ga": 2.26, "gb": 1.5}
            self.gamma_dict["y2"] = {"ga": 3.62, "gb": 0.8}
        else:
            self.gamma_dict = gamma_dict

    def partition(self, D, L):
        """
        Get the
        TODO@DR: not sure if need a seed here for
        in-context variation task to task"""

        return torch.randperm(D).view(L, -1)

    def sample_xs(self, seeds=None):
        """w is the features we construct signal to noise for structure

        y is  from the classification task in jelassi paper.

        TODO@DR: maybe control gamma noise param as a function of y
        instead of the group index.

        Not sure yet how to make in context
        """

        # make a feature here; the signal in the dataset
        w = torch.randn(self.d)
        w /= w.square().sum().sqrt()

        y = torch.randn(self.N).sign()  # this
        # leave it
        q = math.log(self.d) / self.D
        sigma = 1 / math.sqrt(self.d)
        noise = torch.randn(self.N, self.D, self.d) * sigma
        X = torch.zeros(self.N, self.D, self.d)

        for i in range(self.N):
            l = torch.randint(self.L, (1,))[0]
            R = self.S[l]
            for j in range(self.D):
                if j in R:
                    X[i][j] = y[i] * w + noise[i][j]
                else:
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
        return X.reshape(self.N, self.D, self.D, self.D), y, w, self.S


dggen = GroupSampler()
dataset, y, w, partition = dggen.sample_xs()
print(dataset.shape)
# print(y.shape) # 1000; 1 y per example which are tied into signal to noise

# TODO@DR Should I tie the gamma noise params to y or to S partition indeces of groups?

# using X[i][j] = y[i] * w + noise[i][j]
# print(y)

print(dataset[0][0])

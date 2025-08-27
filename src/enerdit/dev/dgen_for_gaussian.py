"""
simple one structure for enerdit dev
"""

import torch
from torch.distributions import Gamma
import torch.nn as nn
import math
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


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

    def add_gaussian_noise(self, X_clean=None, y=None):
        if X_clean is None:  # should come in pair with y
            X_clean, y, w, _ = self.sample_xs()
        X_gnoisy = torch.zeros_like(X_clean)

        # FOr Enerdit N(0,1)

        # TODO@DR I'll have to reason how the seeds for generators go here
        for i in range(X_clean.shape[0]):
            gnoise = torch.randn(1)

            print(f"am I get a z? {gnoise}")

            # gaussian is additive but I need the sqrt here
            noisy_patches = X_clean[i, :, :] + gnoise.view(-1, 1)
            # print(f"clean patch vs noisy patch {X_clean[i, :, :]==noisy_patches}")

            # TODO@DR I'm taking off the noise for right now to debug
            # space-time losses
            X_gnoisy[i, :, :] += noisy_patches
            # X_gnoisy[i, :, :] += X_clean[i, :, :]

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
            # X_clean, X_dirty, y = self.add_gamma_noise()
            X_clean, X_dirty, y = self.add_gaussian_noise()

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

        # TODO@DR: recheck this padding logic, seems something is
        # different as I turn the noise off.

        # want to have dirty first in fused
        # print(f"check that the noise is turned off {X_clean == X_dirty}")

        fused_seq = torch.cat((X_dirty, X_clean), dim=-1)

        # print(f"are dirty and clean diff? {X_dirty==X_clean}") #yes they are different

        return fused_seq, label

    def get_X_and_label_unfused(self, X_clean=None):
        """only the x of clean and the label which is the last of x"""

        patch_dim = X_clean.shape[-1]
        label = torch.zeros((X_clean.shape[0], X_clean.shape[-1] * 2))
        # the label already has double size
        label[:, :patch_dim] += X_clean[:, -1]

        X_clean[
            :, -1
        ] = 0.0  # 0 out last patch where the query is on the clean supervision

        return X_clean, label


# Ad hoc testing
dggen = GroupSampler()
dataset, y, w, partition = dggen.sample_xs()
print(dataset.shape, y.shape)

X, Label = dggen.get_X_and_label_unfused(dataset)
print(f"X.shape {X.shape}")

##########Get batches first


class DatasetWrapper(Dataset):
    """
    (relic): currently, there is a "remainder" batch at the end, with size smaller than batch_size -- could discard it
    """

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

        self.dim_n = self.x.shape[1]
        self.context_length = self.x.shape[2]

    # Mandatory: Get input pair for training
    def __getitem__(self, idx):
        return self.x[idx, :, :], self.y[idx, :]

    # Mandatory: Number of elements in dataset (i.e. size of batch dimension 0)
    def __len__(self):
        X_len = self.x.shape[0]
        return X_len


def grouped_data_train_test_split_util(
    x_total,
    y_total,
    test_ratio,
    as_torch=True,
    rng=None,
):
    # Note that the datagen returns shape (batch, seq len, fused patch dim)
    # the models so far expected a permuted version
    x_total = torch.permute(x_total, (0, 2, 1))

    x_total1 = x_total.numpy()  # these come in as tensors
    y_total1 = y_total.numpy()

    rng = (
        rng or np.random.default_rng()
    )  # determines how dataset is split; if no rng passed, create one

    # now perform train test split and randomize
    ntotal = len(y_total)

    ntest = int(test_ratio * ntotal)
    ntrain = ntotal - ntest

    test_indices = rng.choice(ntotal, ntest, replace=False)
    train_indices_to_shuffle = [i for i in range(ntotal) if i not in test_indices]
    train_indices = rng.choice(train_indices_to_shuffle, ntrain, replace=False)

    # grab train data
    x_train = x_total1[train_indices, :, :]
    y_train = y_total1[train_indices, :]
    # grab test data
    x_test = x_total1[test_indices, :, :]
    y_test = y_total1[test_indices, :]

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


# Get train and test TODO@DR reason why split train test it this way vs generate
# a separate test dataset like in jelassi for the grouped case

x_train, y_train, x_test, y_test = grouped_data_train_test_split_util(
    X, Label, 0.2, as_torch=True, rng=None
)

# print(x_train.shape)
# print(y_train.shape)
batch_size = 3
train_size = x_train.shape[0]

train_dataset = DatasetWrapper(x_train, y_train)
test_dataset = DatasetWrapper(x_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)


def normalize(inputs, target):
    in_min, in_max = torch.min(inputs), torch.max(inputs)
    target_min, target_max = torch.min(target), torch.max(target)
    range = in_max - in_min
    # make it safe
    return (inputs - in_min) / (range + torch.finfo(inputs.dtype).eps), (
        target - target_min
    ) / (torch.finfo(target.dtype).eps + target_max - target_min)


for i, data in enumerate(train_loader):
    inputs, target = data
    print(f"inputs shape {inputs.shape} and target shape {target.shape}")
    # before noise
    # inputs shape torch.Size([3, 64, 8]) and target shape torch.Size([3, 128])

    # normalize to [0, 1]
    inputs, target = normalize(inputs, target)
    b, pdim, seq_len = inputs.shape

    # THis will be the t to generate noise for the seq and to use in loss and in time embed
    t = torch.exp(
        torch.empty(seq_len).uniform_(math.log(10 ** (-9)), math.log(10**3))
    )

    # get z for this batch
    z = torch.randn_like(inputs)
    print(f"z shape {z.shape}")

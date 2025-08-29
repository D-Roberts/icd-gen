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

    def partition(self, D, L, seed=42):
        """
        Get the
        TODO@DR: if I want every S set different, I must change the seed here.
        in-context variation task to task
        - this might be working ok already but double check.
        """
        # torch.manual_seed(seed)

        return torch.randperm(D).view(L, -1)

    # def sample_xs(self, seeds=None):
    #     """w is the features we construct signal to noise for structure

    #     y is  from the classification task in jelassi paper.

    #     Control gamma noise param as a function of y.
    #     Instead could depend on the group index.

    #     On enerdit - put gaussian noise in instead TODO@DR: reason how
    #     this needs to change.

    #     Not sure yet how to best make in context, various options.
    #     Batch context with each batch a str seems apt.
    #     """

    #     # make a feature here; the signal in the dataset
    #     w = torch.randn(self.d)
    #     w /= w.square().sum().sqrt()

    #     y = torch.randn(
    #         self.N
    #     ).sign()  # this is what was the classification label in jelassi, 1 or -1, per instance
    #     # leave it
    #     q = math.log(self.d) / self.D
    #     sigma = 1 / math.sqrt(self.d)
    #     noise = (
    #         torch.randn(self.N, self.D, self.d) * sigma
    #     )  # gaussian noise with this sigma
    #     X = torch.zeros(self.N, self.D, self.d)

    #     for i in range(self.N):  # every example
    #         l = torch.randint(self.L, (1,))[0]  # a group index
    #         # print("l ", l)
    #         # print("partition", self.S)  #
    #         R = self.S[l]
    #         for j in range(self.D):
    #             if (
    #                 j in R
    #             ):  # this here creates artif groups by snr; X[i][j] is meant to be the pixel
    #                 X[i][j] = y[i] * w + noise[i][j]
    #             else:  # TODO@DR: rethink this SNR
    #                 prob = 2 * (torch.rand(1) - 0.5)
    #                 if prob > 0 and prob < q / 2:
    #                     delta = 1
    #                 elif prob < 0 and prob > -q / 2:
    #                     delta = -1
    #                 else:
    #                     delta = 0
    #                 X[i][j] = delta * w + noise[i][j]

    #     # TODO@DR: I might need to rethink how the patches in one sample are
    #     # and how in context is defined

    #     # last dim can be viewed as a patch flattened
    #     return X, y, w, self.S

    def sample_simple(self, d=1024, n=100000, seeds=None):
        """the simplest sampling syntethic for enerdit learning debug.
        d is dimension of patch flattened say 32x32 images

        100,000 samples

        put no context prompt on
        From N(0, 16Id) so var = 16, stdev = 4
        """
        torch.manual_seed(0)
        X = 4 * torch.randn(n, d)

        return X, None, None, self.S  # leave just so that I don't change all interfaces

    def get_X_and_label_unfused(self, X_clean=None):
        """only the x of clean and the label which is the last of x"""

        patch_dim = X_clean.shape[-1]
        print(f"when getting target X clean shape {X_clean.shape}")  # 20, 8, 64
        label = torch.zeros((X_clean.shape[0], X_clean.shape[-1] * 2))
        # the label already has double size
        print(f"when getting target label init shape {label.shape}")

        label[:, :patch_dim] += X_clean[:, -1, :]

        print(f"what was X_clean[:, -1]{X_clean[:, -1].shape}")

        return X_clean, label


# Ad hoc testing
dggen = GroupSampler()
# The simplest Normal
dataset, y, w, partition = dggen.sample_simple()
print(f"shape of simple {dataset.shape}")

# dataset, y, w, partition = dggen.sample_xs()
# print(dataset.shape, y.shape)

# X, Label = dggen.get_X_and_label_unfused(dataset)
# print(f"X.shape {X.shape} and label {Label.shape}")  # 20, 8, 64

##########Get batches first


class DatasetWrapper(Dataset):
    """ """

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

        self.dim_n = self.x.shape[1]
        # self.context_length = self.x.shape[2] # no context for simple

    # Mandatory: Get input pair for training
    def __getitem__(self, idx):
        # return self.x[idx, :, :], self.y[idx, :]
        return self.x[idx, :], self.y[idx]  # FOr simple dummy y

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

    # TODO@DR recall this is for simple so no context
    # x_total = torch.permute(x_total, (0, 2, 1))

    y_total = torch.zeros(x_total.shape[0])  # Dummy for simple

    x_total1 = x_total.numpy()  # these come in as tensors
    y_total1 = y_total.numpy()

    rng = (
        rng or np.random.default_rng()
    )  # determines how dataset is split; if no rng passed, create one

    # now perform train test split and randomize
    ntotal = len(x_total)

    ntest = int(test_ratio * ntotal)
    ntrain = ntotal - ntest

    test_indices = rng.choice(ntotal, ntest, replace=False)
    train_indices_to_shuffle = [i for i in range(ntotal) if i not in test_indices]
    train_indices = rng.choice(train_indices_to_shuffle, ntrain, replace=False)

    # grab train data
    # x_train = x_total1[train_indices, :, :]
    x_train = x_total1[train_indices, :]  # for simple

    y_train = y_total1[train_indices]
    # grab test data
    x_test = x_total1[test_indices, :]  # for simple no context
    x_test = x_total1[test_indices, :]
    y_test = y_total1[test_indices]

    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),  # No y when sampling simple
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


# Get train and test TODO@DR reason why split train test it this way vs generate
# a separate test dataset like in jelassi for the grouped case

# When sampling simple
# x_train, y_train, x_test, y_test = grouped_data_train_test_split_util(
#     dataset, None, 0.2, as_torch=True, rng=None
# )
x_train, y_train, x_test, y_test = grouped_data_train_test_split_util(
    dataset, None, 0.2, as_torch=True, rng=None
)

# print(f"x train shape {x_train.shape}") #b, patchdim, seqlen
# print(f"x train last token {x_train[0, :, -1]}")
# print(f"label {Label}")

# print(y_train.shape)
batch_size = 5  # aim for 512 but debug 5
train_size = x_train.shape[0]

train_dataset = DatasetWrapper(x_train, y_train)
test_dataset = DatasetWrapper(x_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
)


def get_fused_sequences(X_clean, X_noisy):
    """this is to fuse batch seq noisy and clean after adding gaussian
    X_clean is a batch of shape (B, patchdim, seqlen)
    """
    # because the last gnoised patch should be the query and
    # its clean version the label, simply set the last clean
    # patch in the fused to 0

    # TODO@DR reason through how this padding affects the loss
    # calculation and the grad and if I need to do sth about it

    patch_dim = X_clean.shape[-1]

    X_cleaner = torch.empty_like(X_clean)  # make copy to zero out last and fuse
    X_cleaner.copy_(X_clean)

    X_cleaner[
        :, :, -1
    ] = 0.0  # 0 out last patch where the query is on the clean supervision

    fused_seq = torch.cat((X_noisy, X_cleaner), dim=1)  # fuse on the patch dim
    # print(f"are noisy and clean diff? {X_noisy==X_clean}") #yes they are different

    # I already have the label
    return fused_seq


def normalize(inputs, target):
    """to 0,1"""
    dim = target.shape[1]
    in_min, in_max = torch.min(inputs), torch.max(inputs)
    target_min, target_max = torch.min(target), torch.max(target)
    all_min, all_max = min(in_min, target_min), max(in_max, target_max)
    range = all_max - all_min
    # make it safe
    target[:, : dim // 2] = (target[:, : dim // 2] - all_min) / (
        torch.finfo(target.dtype).eps + range
    )
    return (inputs - all_min) / (range + torch.finfo(inputs.dtype).eps), target


def get_batch_samples(data):
    inputs, target = data
    # print(f"the inputs last otken chekc {inputs[2, :, -1]}") # i think as expected
    # normalize to [0, 1]
    # inputs, target = normalize(inputs, target) # Skip for simple

    # b, pdim, seq_len = inputs.shape # no context in simple
    b, pdim = inputs.shape

    # Sample time steps
    # tmin = torch.tensor(10 ** (-9))
    tmin = torch.tensor(0.01)
    # Change the noise tmax considering how small is d
    tmax = torch.tensor(100)

    logtmin = torch.log(tmin)
    logtmax = torch.log(tmax)

    logt_distrib = Uniform(low=torch.tensor([logtmin]), high=torch.tensor([logtmax]))
    logt = logt_distrib.sample(torch.tensor([b]))
    t = torch.exp(logt).squeeze(-1)  # like 0.15 to 227 etc
    # print(f"generated t is {t} and shape {t.shape}")
    # print(f"the t {t}")

    # get z for this batch from N(0,I)
    z = torch.randn_like(inputs)
    # print(f"z shape {z.shape}")
    sqrttz = torch.zeros_like(z)

    # I am applying the same t noise accross the sequence in one instance
    # and diffeernt t accross the minibatch

    # sqrttz = torch.einsum('bcd,b->bd', z, torch.sqrt(t))
    sqrttz = torch.einsum("bd,b->bd", z, torch.sqrt(t))  # For simple only 2-dim

    # print(f"the noise*sqrtt last token {sqrttz[0,:,-1]}")

    # test that the broadcasting happened as expected
    # print(f" check {sqrttz[0,1,0] / sqrtt[0]} and {z[0, 1, 0]}") #ok

    # Get noisy seq for the batch
    noisy = torch.zeros_like(inputs)
    noisy += inputs
    noisy += sqrttz

    # print(
    #     f"what does noisy look like {torch.mean(noisy), torch.var(noisy), noisy.shape}"
    # )

    # Get fused seq for the batch; query is last; noisy first

    # No fusing no prompt on simple
    # fused = get_fused_sequences(inputs, noisy)  # this is a batch
    fused = None

    # print(f"the fused shape {fused.shape}") # ok double patch dim
    # print(f"the fused last otken chekc {fused[2,:,-1]}")
    # print(f"the noisy last otken chekc {noisy[2,:,-1]}") # i think as expected

    # so now have inputs (clean), target, z, noisy only, fused, t

    # return t, z, target, fused
    return t, z, inputs, noisy  # for simple


# for i, data in enumerate(train_loader):
#     t, z, target, xs = get_batch_samples(data)
#     # print(f"inputs shape {inputs.shape} and target shape {target.shape}")

#     # print(f"returned z.shape {z.shape}") #(B, patc, seq)
#     # print(f"returned xs.shape {xs.shape}") #(B, 2patc, seq)
#     # print(t) # a seq len
#     # print(f"target shape {target.shape}") #(B, 2patch)
#     # print(f"the fused last otken chekc {xs[2,:,-1]}")
#     # print(f"the target last otken chekc {target[2]}") # i think as expected

#     break

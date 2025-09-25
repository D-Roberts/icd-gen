"""
Data generation module for inputs with groups. Noising with gamma for
the in-context sequence.

@DR: note that there is a torch.distributions.exp_family.ExponentialFamily
abstract class to possibly work with in a more flexible/general datagen way
beyond normal.

# note that gamma noise is always positive; it is multiplicative noise
# gamma distribution is in the exponential family
# To add gamma noise to an image, you would typically multiply the image by the sampled noise
# X= VY where V is gamma noise

# THe prompt and query will be (y1,x1,y2,x2,y3,x3,y4) and predict x4- clean
# to construct train and test dataset, label will be x4.
# where y are the noised; we aim to learn distribution through noise-clean
# associations as well as input group structure; for this - must have positions
# encoded.
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
        N=5000,  # small for dev
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
        TODO@DR: not sure about the see here or not for in-context"""
        # torch.manual_seed(seed)

        return torch.randperm(D).view(L, -1)

    def sample_xs(self, seeds=None):
        """w is the features we construct signal to noise for structure

        y is  from the classification task in jelassi paper.

        Control gamma noise param as a function of y. Instead could depend on the group index.

        Not sure yet how to best make in context, various options. Batch context
        seems apt as well.
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
        # and how the groups - gammas work
        count_gamma1 = 0
        count_gamma2 = 0
        for i in range(X_clean.shape[0]):
            if y[i] == 1:
                gnoise = gamma_dist1.sample((1,))
                count_gamma1 += 1
            else:  # -1
                gnoise = gamma_dist2.sample((1,))
                count_gamma2 += 1

            # gamma is multiplicative
            noisy_patches = X_clean[i, :, :] * gnoise.view(-1, 1)
            X_gnoisy[i, :, :] += noisy_patches

        # so each instance will have noise from either gamma1 or gamma2 randomly as y is drawn but related to
        # how the underlying data is generated. Not sure if I want to do this or not.

        print(f"how many times gamma1 {count_gamma1} and gamma2 {count_gamma2}")
        return X_clean, X_gnoisy, y

    def get_fused_sequence(self, X_clean=None, X_corrupted=None):
        # just assume if X_clean given we also have X_corrupted

        # TODO@dr I am calling it corrupted bc I wonder about other distortions
        # beside noise; for instance those eigendistortions. Can
        # I do anything with that?

        if X_clean is None:
            X_clean, X_corrupted, y = self.add_gamma_noise()

        # because the last gnoised patch should be the query and
        # its clean version the label, simply set the last clean
        # patch in the fused to 0 TODO@DR reason through how this affects the mean loss
        # calculation and the grad

        patch_dim = X_clean.shape[-1]
        label = torch.zeros((X_clean.shape[0], X_clean.shape[-1] * 2))
        label[:, :patch_dim] += X_clean[:, -1]

        X_clean[
            :, -1
        ] = 0.0  # 0 out last patch where the query is on the clean supervision

        # want to have corrupted first in fused
        fused_seq = torch.cat((X_corrupted, X_clean), dim=-1)

        # print(f"are corrupted and clean diff? {X_corrupted != X_clean}") #yes they are different

        return fused_seq, label


# Ad hoc testing
dggen = GroupSampler()
dataset, y, w, partition = dggen.sample_xs()
# torch.save(dataset, 'group_data/dataset.to')
# torch.save(y, 'group_data/y.to')
# torch.save(w, 'group_data/w.to')
# torch.save(partition, 'group_data/partition.to')

# dataset = torch.load("group_data/dataset.to")
# y = torch.load("group_data/y.to")
# w = torch.load("group_data/w.to")
# partition = torch.load("group_data/partition.to")
# print(f"partition loaded now {partition}")
# print(f"shape of w {w.shape}")


# dataset[:,-1] = 0.0
# print("last patch?", dataset[0,-1].shape) # yeap
# print("first sample with last patch zeroed", dataset[0]) #yeap

# print(y.shape) # 1000; 1 y per example which are tied into signal to noise
# TODO@DR Should I tie the gamma noise params to y or to S partition indeces of groups?

_, noisy_d, _ = dggen.add_gamma_noise(dataset, y)
# print(f"the noisy set {noisy_d.shape}")  # (num_samples, num_patches, flattened patch)
fused_seq, label = dggen.get_fused_sequence(dataset, noisy_d)
# print(f"label shape {label.shape} and val {label[0]}") # looks ok now
# print(f"check that last fused patch in teh seq has val 0 {fused_seq[0][-1].shape} and not equal to label {fused_seq[0][-1]} at the same pos") #yeap
# print(f"fused seq shape {fused_seq.shape}")
# print(f"feature w shape {w.shape}") #100 dim vector from d
# print(f"partition of indices S {partition}") # there are 10 groups with 2 elem each

# TODO@DR: What does this mean? Rethink how the groups and group indeces work.


def plot_the_batch_partitions(i, D, partition):
    plt.figure(figsize=(4, 4))
    S_matrix = torch.eye(D, D)
    for x in partition:
        # print(f"wjat is x in S {x[1]}")
        S_matrix[x[0], x[1]] = 1 / 2
        S_matrix[x[1], x[0]] = 1 / 2
    plt.matshow(S_matrix, cmap="Greys")
    plt.axis("off")
    plt.savefig(f"batch_groups_png/batch{i}.png")


# POC Batch group for In-Batched-Context Learning
# a new structure / distribution for each batch; since noise is tied to
# the labels y which are tied to the data distribution - a new noise distrib
# TODO @DR will have to ascertain this


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
    fused_seq, label, 0.2, as_torch=True, rng=None
)

print(f"train X {x_train.shape}")
print(f"train y {y_train.shape}")


############################################
# **************************************Get batch groups for One In-Context Learning Style Training
from torch.utils.data import Dataset, DataLoader


# Custom class for an already batched dataset
class PreBatchedDataset(Dataset):
    def __init__(self, batched_data):
        self.batched_data = batched_data

    def __len__(self):
        return len(self.batched_data)

    def __getitem__(self, idx):
        # Returns an already batched sample
        return self.batched_data[idx]


# Train and test separatelly makes sense


# num_batches_train = 100  # when a real number of batches - this takes a while
# batch_size = 128
# num_batches_test = 50
# D = 10


# def get_batch_groups(num_batches, N=batch_size):
#     # TODO@DR this should have
#     # all parameters of generator;
#     train_set = []

#     for i in range(num_batches_train):
#         # different structure on each batch of 128 instances
#         dggen = GroupSampler(N=N, D=10)
#         dataset, y, w, partition = dggen.sample_xs()

#         # print(
#         #     "partition", partition
#         # )  # TODO@DR - double check if I want different groups and partitions
#         # and then if this code does this like I want

#         # Each batch will have a different partition as plots show
#         # Effectively a different structure; can think of the batch
#         # as one image

#         # plot_the_batch_partitions(i=i, D=10, partition=partition)

#         _, noisy_d, _ = dggen.add_gamma_noise(dataset, y)
#         fused_seq, label = dggen.get_fused_sequence(dataset, noisy_d)
#         X = torch.permute(fused_seq, (0, 2, 1))
#         train_set.append((X, label))

#     return train_set


# train_batched_data = get_batch_groups(num_batches=num_batches_train, N=batch_size)
# test_batched_data = get_batch_groups(
#     num_batches_test, batch_size
# )  # the case with groups per batch


# # Adhoc test ****************************************
# dataset = PreBatchedDataset(train_batched_data)

# Initialize the DataLoader with the custom Dataset.
# Crucially, set batch_size=1 because each item returned by
# __getitem__ is already a full batch. Also, set shuffle=False and
# collate_fn=None (or a simple identity function) as no further
# batching or collation is needed.
# train_loader = DataLoader(dataset, batch_size=1, shuffle=False,
#                          collate_fn=lambda x: x[0])

# test_loader = DataLoader(
#     dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
# )


# for batch_idx, batch in enumerate(test_loader):
#     features, labels = batch
#     print(
#         f"Batch {batch_idx}: Features shape {features.shape}, Labels shape {labels.shape}"
#     )


# **********************************************************
# AdHoc test ***********************************************
# Look at nlmeans and psnr on this generated dataset

# # estimate the noise standard deviation from the noisy image
# clean = dataset[0,0].view(10,10).unsqueeze(-1).numpy()
# noisy = noisy_d[0,0].view(10,10).unsqueeze(-1).numpy()
# print(noisy.shape)

# # print(noisy.shape)
# sigma_est = np.mean(estimate_sigma(noisy, channel_axis=-1))
# print(f'estimated noise standard deviation = {sigma_est}')

# patch_kw = dict(
#     patch_size=5,  # 5x5 patches
#     patch_distance=6,  # 13x13 search area
#     channel_axis=-1,
# )

# denoise2_fast = denoise_nl_means(
#     noisy, h=0.6 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw
# )

# print("den fast shape", denoise2_fast.reshape((10, 10, 1)).shape)
# denoise2_fast = denoise2_fast.reshape((10, 10, 1))

# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 6), sharex=True, sharey=True)

# ax[0].imshow(noisy)
# ax[0].axis('off')
# ax[0].set_title('noisy')
# ax[1].imshow(clean)
# ax[1].axis('off')
# ax[1].set_title('clean')
# ax[0, 2].imshow(denoise2)
# ax[0, 2].axis('off')
# ax[0, 2].set_title('non-local means\n(slow, using $\\sigma_{est}$)')
# ax[1, 0].imshow(astro)
# ax[1, 0].axis('off')
# ax[1, 0].set_title('original\n(noise free)')
# ax[1, 1].imshow(denoise_fast)
# ax[1, 1].axis('off')
# ax[1, 1].set_title('non-local means\n(fast)')
# ax[2].imshow(denoise2_fast)
# ax[2].axis('off')
# ax[2].set_title('non-local means\n(fast, using $\\sigma_{est}$)')

# fig.tight_layout()
# plt.show()

# print PSNR metric for each case; higher is better (closer img) but using MSE
# minv = min(np.min(noisy), np.min(clean))
# maxv = max(np.max(noisy), np.max(clean))

# psnr_noisy = peak_signal_noise_ratio(clean, noisy)
# ssim_noisy = structural_similarity(im1=clean, im2=noisy,
#                                    gaussian_weights=True,
#                                    data_range=maxv-minv, sigma=1.5, win_size=1,
#                                    use_sample_covariance=False)

# psnr2_fast = peak_signal_noise_ratio(clean, denoise2_fast) # inf for same image

# minv = min(np.min(denoise2_fast), np.min(clean))
# maxv = max(np.max(denoise2_fast), np.max(clean))

# ssim_nlmeans = structural_similarity(im1=clean, im2=denoise2_fast,
#                                    gaussian_weights=True,
#                                    data_range=maxv-minv, sigma=1.5,
#                                    win_size=1,
#                                    use_sample_covariance=False)


# print(f'PSNR (noisy) = {psnr_noisy:0.2f}')
# print(f'PSNR (fast, using sigma est) = {psnr2_fast:0.2f}')

# print(f'SSIM (noisy) = {ssim_noisy:0.2f}')
# print(f'SSIM (fast, using sigma est) = {ssim_nlmeans:0.2f}') #1 for same image
# higher so nlmeans does do some denoising on the gamma

# SSIM is a better metric for this structure and gamma case

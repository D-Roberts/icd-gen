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

    def patchify(self, im_tensor, patch_size=(32, 32), stride=(32, 32)):
        """
        im_tensor: (batch_size, chan, im_dim, im_dim)
        """

        unfold = nn.Unfold(
            kernel_size=patch_size, stride=stride
        )  # not sure if I need padding here or not
        patches = unfold(im_tensor)
        return patches


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


class GaussianSampler(DataSampler):
    """Generate one image-like tensor to patchify later.
    Args:
        im_dim - int - image dimension (assuming square images for now)
        mean - float - mean of the gaussian
        sigma - float - std.dev of the gaussian
    Returns:
        samples - torch.Tensor - sampled images of shape (num_samples, im_dim*im_dim)


    """

    def __init__(self, im_dim, sigma=1.0):
        """for now have an identity covariance scaled by sigma"""
        super().__init__()
        self.mean = torch.zeros((im_dim, im_dim))  # assume 0 mean for now
        # sample uniform later
        self.sigma = sigma
        self.im_dim = im_dim

    def sample_xs(self, b_size, seeds=None):
        """bsize is the batch size here.
        1 seed for each batch element.
        """
        cov_mat = torch.eye(int(self.im_dim)) * self.sigma
        if seeds is None:
            # Create the MultivariateNormal distribution
            mvn = MultivariateNormal(loc=self.mean, covariance_matrix=cov_mat)

            # Sample from the distribution
            # To get 10 samples, pass a sample_shape of (10,)
            samples = mvn.sample((b_size,))
        else:
            samples = torch.zeros(b_size, self.im_dim, self.im_dim)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                # not sure this takes it right
                generator.manual_seed(seed)
                # Create the MultivariateNormal distribution
                mvn = MultivariateNormal(loc=self.mean, covariance_matrix=cov_mat)
                samples[i] = mvn.sample((1,))

        return samples


# Ad-hoc Test
b_size = 3
im_size = 32
patch_dim = 16
gamma_alpha = 2.0
gamma_beta = 1.0

im_gen = GaussianSampler(im_size)
d1 = im_gen.sample_xs(b_size=b_size, seeds=[0, 1, 2])
print(d1.shape)
# reshape im_tensor to (batch_size, channels, height, width)
d1_reshaped = d1.unsqueeze(1)

patches = im_gen.patchify(
    d1_reshaped, patch_size=(patch_dim, patch_dim), stride=(patch_dim, patch_dim)
)  # assume square patches non-overlapping from
# square image
print(patches.shape)
# To reshape the patches into a more intuitive format
#  (e.g., [batch_size, num_patches, channels, patch_height, patch_width]),
# you can use view and potentially permute:

reshaped_patches = patches.view(
    d1_reshaped.shape[0],  # batch_size
    -1,  # Infer the number of patches
    d1_reshaped.shape[1],  # channels
    patch_dim,  # patch_height
    patch_dim,  # patch_width
)

print(f"Shape after reshaping: {reshaped_patches.shape}")

# I am assuming grey scale and remove the channel dim until needed
patches = reshaped_patches.squeeze(2)

print(patches.shape)  # (batch, num_p, patch_dim, patch_dim)
# print(patches) # 3 batches with 4 patches in total each
# This is the sequence of patches from topleft to bottomright

# --------------------------
# add some gamma noise to all patches to get Ys sequence;
# this is multiplicative noise.
# Define the parameters of the Gamma distribution
concentration_param = torch.tensor([gamma_alpha])  # Alpha (shape parameter)
rate_param = torch.tensor([gamma_beta])  # Beta (rate parameter)

# Create a Gamma distribution object
gamma_dist = Gamma(concentration=concentration_param, rate=rate_param)

# Sample from the distribution
# You can specify the number of samples you want to draw
noise_samples = gamma_dist.sample((b_size,))

# print(f"Gamma samples: {samples}") # works
# note that gamma noise is always positive; it is multiplicative noise
# gamma distribution is in the exponential family
# To add gamma noise to an image, you would typically multiply the image by the sampled noise
# X= VY where V is gamma noise
# For example, if 'image_tensor' is your image in tensor form:
# noisy_image = image_tensor * samples.view(-1, 1, 1)
# Make sure to adjust the shape of 'samples' to match the dimensions of 'image_tensor.

# noise all the patches to get labels
noisy_seq = patches * noise_samples.view(-1, 1, 1, 1)
print(noisy_seq.shape)

# THe prompt and query will be (y1,x1,y2,x2,y3,x3,y4) and predict x4- clean
# to construct train and test dataset, label will be x4.
# where y are the noised; we aim to learn distribution through noise-clean
# associations as well as patch structure; for this - must have positions

# concatenate clean and noisy patches into sequence of concatenated
# shape (batchsize, num_patches, patch_dim, 2*patch_dim)
instance = torch.cat((noisy_seq, patches), dim=-1)
print(instance.shape)


# TODO@DR consider getting overlapping patches more similarly to NL Means


# An example patch embedding - will change later
# now the sequence has double width due to concat clean and noisy
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        # aim to embed the clean and noisy together
        # set bias to zero
        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=(patch_size, 2 * patch_size),
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            device=None,
            dtype=None,
        )
        self.num_patches = num_patches

    def forward(self, x):
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # TODO@DR check for correctness again
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # print("torch.sin(position * div_term)", torch.sin(position * div_term).shape)
        # print("pe[:, 0::2]", pe[:, 0::2].shape)
        # print("pe[:, 1::2]", pe[:, 1::2].shape)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # print("pe shape", pe.shape)
        self.register_buffer("pe", pe.unsqueeze(0))  # Add a batch dimension

    def forward(self, x):
        # x is the sequence of patch embeddings, shape (batch_size, seq_len, d_model)
        # We need to add positional embeddings to each element in the sequence
        # for the moment only create the pos embeddings

        return self.pe[:, : x.size(1)]


# Either this or simply time step embedding
pos_embed_dim = 8
pos_embed = SinusoidalPositionalEmbedding(pos_embed_dim, max_len=instance.shape[1])(
    instance
)
# print("pos embed shape", pos_embed.shape) # first dim is batch, the time dim / pos is the same for each batch

# these will be added to the image patch embedding

patch_embedder = PatchEmbedding(
    embed_dim=8, patch_size=16, num_patches=instance.shape[1]
)
# reshape into a shape of image with w double h
embed_patches = patch_embedder(instance.reshape(b_size, 1, im_size, 2 * im_size))

# print(embed_patches.shape) #(batch, num_patches, embed_dim)

# Then we would add the embeddings together
input_embed = pos_embed + embed_patches
print(input_embed.shape)  # (batc, num patches or seq len of fused patches, embed dim)


# ********************** then I'd denoise the query only (the query is given noised)
# ********************** Also test on pure noise query to see what happens.

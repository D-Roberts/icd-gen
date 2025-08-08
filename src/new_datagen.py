"""
Patch generation module for new datagen strategies.

@DR: note that there is a torch.distributions.exp_family.ExponentialFamily 
abstract class to possibly work with in a more flexible/general datagen way
beyond normal.

"""
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn


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
im_gen = GaussianSampler(32)
d1 = im_gen.sample_xs(b_size=3, seeds=[0, 1, 2])
print(d1.shape)
# reshape im_tensor to (batch_size, channels, height, width)
d1_reshaped = d1.unsqueeze(1)
patch_dim = 16
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

print(patches.shape)
# print(patches) # 3 batches with 4 patches each

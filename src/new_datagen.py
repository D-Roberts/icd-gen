"""
Patch generation module for new datagen strategies.

@DR: note that there is a torch.distributions.exp_family.ExponentialFamily 
abstract class to possibly work with in a more flexible/general datagen way
beyond normal.

"""
import torch
from torch.distributions import MultivariateNormal


class DataSampler:
    def __init__(self):
        pass

    def sample_xs(self):
        raise NotImplementedError


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
print(im_gen.sample_xs(b_size=1, seeds=[1]))

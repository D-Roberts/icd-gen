import os
import sys

from random import randint
import uuid

import argparse

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax as scipy_softmax

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler


from models import build_model, TransformerModelV1nores, TransformerModelV3, TransformerModelV2nores

import wandb

torch.backends.cudnn.benchmark = True
if not torch.backends.mps.is_available():
    print('\nMPS device not found.')
    mps_device = None
     
if torch.backends.mps.is_available():
        device = torch.device("mps")
        mps_device = torch.device("mps")
        x = torch.ones(1, device=device)
        print('\nCheck M1 chip:', x)
elif torch.cuda.is_available():
        device = torch.device("cuda:0")
else:
        device = "cpu"
print('device selected:', device)

#from denoise
def report_dataset_loss(net, criterion, dataloader, data_label, print_val=False, plot_some=False):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data
            outputs = net(samples.to(device))

            if plot_some:
                # visualization and debugging
                if count < 5:
                    plt.plot(outputs, label='pred')
                    plt.plot(targets, label='true')
                    plt.legend()
                    plt.show()

            mse_loss_total += criterion(outputs.to(device), targets.to(device)).item()  # add the *mean* MSE for the batch to mse_loss_total


            # Note: we could do count += bsz, but we assume batch-mean inside criterion call
            # e.g. count += samples.size(0)
            #  - this would resolve edge issue when dataset (e.g. test set) not divisible by bsz
            count += 1  # this +1 and divide by count assumes batch-mean inside criterion call, as is done above

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss: %.3e (batches=%d)' % (data_label, mse_loss, count))
    return mse_loss

DIR_CURRENT = os.path.dirname(__file__)
DIR_PARENT = os.path.dirname(DIR_CURRENT)
sys.path.append(DIR_PARENT)


# default kwargs for dataset generation for each case 0: 'Linear', 1: 'Clustering', 2: 'Manifold'
DATAGEN_GLOBALS = {
    0: dict(
        sigma2_corruption=0.5,                    # applies to all cases
        style_corruption_orthog=False,            # if True, the noise is only in orthogonal directions
        style_origin_subspace=True,               # if True, the subspace must contain origin (not affine)
        style_subspace_dimensions='random',       # int or 'random'
        # parameters specific to the Linear case
        sigma2_pure_context=2.0,                  # controls radius of ball of pure of samples (default: 2.0)
        corr_scaling_matrix=None,                 # x = Pz; y = A x  \tilde x = A (x + z); (normally identity I_n)
    ),
    1: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,     # mandatory for this case
        style_origin_subspace=True,        # mandatory for this case
        style_subspace_dimensions='full',  # full means dim-n gaussian balls in R^n
        # parameters specific to the Clustering case
        style_cluster_mus='unit_norm',
        style_cluster_vars='isotropic',
        num_cluster=3,
        cluster_var=0.01,                   # controls radius of ball of pure of samples (default: 2.0); could be a list in Clustering case
    ),
    2: dict(
        sigma2_corruption=0.5,                # applies to all cases
        style_corruption_orthog=False,        # mandatory for this case
        style_origin_subspace=True,           # mandatory for this case
        style_subspace_dimensions='random',   # int or 'random'
        # parameters specific to the Manifold case
        radius_sphere=1.0,                    # controls radius of sphere (d-dim manifold S in R^n)
    ),
}

# asserts for Linear case
assert DATAGEN_GLOBALS[0]['style_subspace_dimensions'] in ['random'] or isinstance(DATAGEN_GLOBALS[0]['style_subspace_dimensions'], int)
assert DATAGEN_GLOBALS[0]['style_corruption_orthog'] is False
assert DATAGEN_GLOBALS[0]['style_origin_subspace'] is True


DIR_OUT = DIR_PARENT + os.sep + 'output'
DIR_DATA = DIR_PARENT + os.sep + 'input'
DIR_MODELS = DIR_PARENT + os.sep + 'models'
DIR_RUNS = DIR_PARENT + os.sep + 'runs'

for core_dir in [DIR_OUT, DIR_DATA, DIR_MODELS, DIR_RUNS]:
    if not os.path.exists(core_dir):
        os.mkdir(core_dir)

def load_runinfo_from_rundir(dir_run, model_prefix=''):
    """
    Load runinfo.txt from a given run directory
    """
    runinfo_fpath = dir_run + os.sep + model_prefix + 'runinfo.txt'
    with open(runinfo_fpath, 'r') as runinfo:
        runinfo_lines = runinfo.readlines()

    # step: convert runinfo to a dictionary and return it
    runinfo_dict = {}
    for line in runinfo_lines:
        if line.split(',')[0].strip() == 'scheduler':
            key = line.split(',')[0]
            val = ','.join(line.split(',')[1:]).strip()
            val = eval(val)
            assert isinstance(val, dict)
        else:
            key, val = line.split(',')
            key, val = key.strip(), val.strip()
        runinfo_dict[key] = val

    # handle potentially ambiguous key -> value types
    # - style_subspace_dimensions (int or str)
    # - seed_dataset (None or int)
    # - could have diff number of keys relating to gradient descent e.g. adam_lr, sgd_lr, etc. -- keep as str
    for key in ['epochs', 'batch_size', 'dim_n', 'context_len',
                'train_plus_test_size', 'full_loss_sample_interval',
                'context_examples_per_W', 'samples_per_context_example', 'num_W_in_dataset']:
        runinfo_dict[key] = int(runinfo_dict[key])
    for key in ['style_subspace_dimensions']:
        if runinfo_dict[key] != 'full':
            runinfo_dict[key] = int(runinfo_dict[key])
    for key in ['sigma2_corruption', 'sigma2_pure_context', 'test_ratio']:
        runinfo_dict[key] = float(runinfo_dict[key])
    for key in ['style_corruption_orthog', 'style_origin_subspace']:
        runinfo_dict[key] = runinfo_dict[key] == 'True'  # if True, the bool val is True, else False

    # specific checks for particular datagen cases
    if 'case1_specific_style_cluster_mus' in runinfo_dict.keys():
        runinfo_dict['style_cluster_mus'] = runinfo_dict['case1_specific_style_cluster_mus']
        del runinfo_dict['case1_specific_style_cluster_mus']

    if 'case1_specific_style_cluster_vars' in runinfo_dict.keys():
        runinfo_dict['style_cluster_vars'] = runinfo_dict['case1_specific_style_cluster_vars']
        del runinfo_dict['case1_specific_style_cluster_vars']

    if 'case1_specific_cluster_var' in runinfo_dict.keys():
        runinfo_dict['cluster_var'] = float(runinfo_dict['case1_specific_cluster_var'])
        del runinfo_dict['case1_specific_cluster_var']

    if 'case1_specific_num_cluster' in runinfo_dict.keys():
        runinfo_dict['num_cluster'] = int(runinfo_dict['case1_specific_num_cluster'])
        del runinfo_dict['case1_specific_num_cluster']
    return runinfo_dict


# asserts for Linear case
assert DATAGEN_GLOBALS[0]['style_subspace_dimensions'] in ['random'] or isinstance(DATAGEN_GLOBALS[0]['style_subspace_dimensions'], int)
assert DATAGEN_GLOBALS[0]['style_corruption_orthog'] is False
assert DATAGEN_GLOBALS[0]['style_origin_subspace'] is True

def gen_uniform_data(size, rng=None):
    """
    Return random np.array ~ *size* with U[-1,1] elements
    - note $U[-1,1]$ has variance $1/12 (b-a)^2$ which gives $1/3$
    - scaling the arr by C increases the variance by C^2 (and thus the std.dev by C)
    """
    rng = rng or np.random.default_rng()  # Use provided RNG or create one
    arr = 2 * rng.random(size=size) - 1
    return arr


def gen_gaussian_data(mean, cov, size, rng=None):
    """
    Return random np.array ~ *size* with N(mu, cov) elements
    - if size is an integer, then returned arr is {size x dim n}
    """
    rng = rng or np.random.default_rng()
    arr = rng.multivariate_normal(mean, cov, size=size, check_valid='raise')  # size x dim n
    return arr

# function to sample L vectors from W = {x in R^n | x = m + V c} for some c in R^d
def sample_from_subspace(W_m, W_V, nsample, sigma_Z_sqr=1.0, uniform=True, rng=None):
    """
    Args
    - W_m - array-like - n x k
    - W_V - array-like - n x k

    For each sample
    - sample c -- U[-1, 1] vector of random coefficients in R^d (for basis W_V)
    - then x = W_m + W_V @ c is a random vector from the subspace
    Return:
        np.array of size (n x nsample) where n is the dimension of the ambient space (n >= dim W == W_V.shape[1])
    """
    rng = rng or np.random.default_rng()
    dim_n, dim_W = W_V.shape
    if uniform:
        c = gen_uniform_data(
            (dim_W, nsample), rng=rng)  # random coefficients for the basis {v_i}
        # note by default, c will have a variance of 1/3 (U[-1,1]); we can scale it by
        c = np.sqrt(3 * sigma_Z_sqr) * c  # e.g. if goal is variance of 10, then scale sqrt(3 * 10)
        rand_samples = np.expand_dims(W_m, axis=1) + W_V @ c
    else:
        # gaussian ball method
        x_blob = rng.multivariate_normal(W_m, sigma_Z_sqr * np.eye(dim_n), nsample)
        projector_exact = W_V @ np.linalg.inv(W_V.T @ W_V) @ W_V.T
        projection_b_onto_W = np.expand_dims(W_m, axis=1) + projector_exact @ (x_blob.T - np.expand_dims(W_m, axis=1))
        rand_samples = projection_b_onto_W
    return rand_samples

def data_train_test_split_util(x_total, y_total, data_subspace_dict, context_len, dim_n, num_W_in_dataset,
                               context_examples_per_W, test_ratio, as_torch=True,
                               savez_fname=None, verbose=True, rng=None):
    """
    Used commonly by data_train_test_split_* functions - for * = {linear, clusters, manifold}

    TODO make it so test ratio 1.0 or 0.0 makes the corresponding array empty (as opposed to nasty if/else currently)
    """
    if verbose:
        print('Generating train/test data for NN training (context length = %d)...' % context_len)
        print('\tcontext_len=%d, dim_n=%d, num_W_in_dataset=%d, examples_per_W=%d, test_ratio=%s' %
              (context_len, dim_n, num_W_in_dataset, context_examples_per_W, test_ratio))
        print('\tnsample_per_subspace (context_len):', context_len)
        print('Total dataset size before split: %d (x,y) pairs' % len(y_total))

    x_total = np.array(x_total).astype(np.float32)
    y_total = np.array(y_total).astype(np.float32)

    rng = rng or np.random.default_rng()  # determines how dataset is split; if no rng passed, create one

    # now perform train test split and randomize
    ntotal = len(y_total)
    if test_ratio is None:
        x_test = None
        y_test = None
        test_data_subspaces = None

        ntrain = ntotal
        train_indices_to_shuffle = [i for i in range(ntotal)]
        train_indices = rng.choice(train_indices_to_shuffle, ntrain, replace=False)

        # grab train data
        x_train = x_total[train_indices, :, :]
        y_train = y_total[train_indices, :]
        # rebuild metadata dicts after shuffling
        train_data_subspaces = dict()
        for idx, val in enumerate(train_indices):
            train_data_subspaces[idx] = data_subspace_dict[val].copy()

        if verbose:
            print('\t x_train:', x_train.shape)
            print('\t y_train:', y_train.shape)
            print('\t x_test: None')
            print('\t y_test: None')

        if savez_fname is not None:
            assert savez_fname[-4:] == '.npz'
            # save dataset to file
            dataset_fpath = savez_fname
            print('\nSaving dataset to file...', dataset_fpath)
            np.savez_compressed(dataset_fpath,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=np.empty(1),
                                y_test=np.empty(1))

        if as_torch:
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)

    else:
        ntest = int(test_ratio * ntotal)
        ntrain = ntotal - ntest

        test_indices = rng.choice(ntotal, ntest, replace=False)
        train_indices_to_shuffle = [i for i in range(ntotal) if i not in test_indices]
        train_indices = rng.choice(train_indices_to_shuffle, ntrain, replace=False)

        # grab train data
        x_train = x_total[train_indices, :, :]
        y_train = y_total[train_indices, :]
        # grab test data
        x_test = x_total[test_indices, :, :]
        y_test = y_total[test_indices, :]

        # rebuild metadata dicts after shuffling
        train_data_subspaces = dict()
        test_data_subspaces = dict()
        for idx, val in enumerate(train_indices):
            train_data_subspaces[idx] = data_subspace_dict[val].copy()
        for idx, val in enumerate(test_indices):
            test_data_subspaces[idx] = data_subspace_dict[val].copy()

        if verbose:
            print('\t x_train:', x_train.shape)
            print('\t y_train:', y_train.shape)
            print('\t x_test:', x_test.shape)
            print('\t y_test:', y_test.shape)

        if savez_fname is not None:
            assert savez_fname[-4:] == '.npz'
            # save dataset to file
            dataset_fpath = savez_fname
            print('\nSaving dataset to file...', dataset_fpath)
            np.savez_compressed(dataset_fpath,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test)

        if as_torch:
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
            x_test = torch.from_numpy(x_test)
            y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces

def data_train_test_split_linear(
        context_len=100,
        dim_n=8, # was 128
        num_W_in_dataset=100,
        context_examples_per_W=1,
        samples_per_context_example=1,
        test_ratio=0.2,
        verbose=True,
        seed=None,
        style_subspace_dimensions=DATAGEN_GLOBALS[0]['style_subspace_dimensions'],
        style_origin_subspace=DATAGEN_GLOBALS[0]['style_origin_subspace'],      # TODO convert to float  0 < m < 1 magnitude
        style_corruption_orthog=DATAGEN_GLOBALS[0]['style_corruption_orthog'],  # TODO convert to float  0 < a < 1
        sigma2_corruption=DATAGEN_GLOBALS[0]['sigma2_corruption'],
        sigma2_pure_context=DATAGEN_GLOBALS[0]['sigma2_pure_context'],
        corr_scaling_matrix=DATAGEN_GLOBALS[0]['corr_scaling_matrix'],
        savez_fname=None,
        as_torch=True):
    """
    Args:
    - test_ratio: float      - 0 <= x <= 1
    - seed: None or int      - static randomization for the order of examples in test and train set

    Vectors x_i live in R^n

    Training sequences are of the form {x_1, ..., x_L, x_query} -> x_{L+1}
     - x_1, ..., x_L lie on affine space W of dimension d < n
     - the final input x_query lies outside W, and the task is to de-corrupt it by projecting onto W
        - corruption procedure A: kick size of random magnitude in a direction orthogonal to W
            kick_size = mu + sigma * np.random.randn()   (mu, sigma both 1.0)
        - corruption procedure B: iid gaussian var = kick_size centered

       W = {x in R^n | x = m + V c} for some c in R^d
        - we do not directly provide V [n x d] or m [n x 1]
        - these can be inferred from the mean and by PCA

    - Teacher solution:
        g(x_q) = mu + P (x_q - mu)  where  P_W = V (V^T V)^-1 V^T defines the projection onto underlying vector space
                                            mu = mean(x_1, ..., x_L)
    """
    print('data_train_test_split_linear() args...')
    print('\tstyle_subspace_dimensions:', style_subspace_dimensions)
    print('\tstyle_origin_subspace:',     style_origin_subspace)
    print('\tstyle_corruption_orthog:',   style_corruption_orthog)
    print('\tcontext_len:', context_len)
    print('\tdim_n:', dim_n)
    print('\tnum_W_in_dataset:', num_W_in_dataset)
    print('\tcontext_examples_per_W:', context_examples_per_W)
    print('\tsamples_per_context_example:', samples_per_context_example)
    print('\tsigma_corruption:', sigma2_corruption)
    print('\tsigma_pure_context:', sigma2_pure_context)
    print('\tcorr_scaling_matrix:', corr_scaling_matrix)
    print('\tseed:', seed)
    print('\tsavez_fname:', savez_fname)

    assert samples_per_context_example == 1

    rng = np.random.default_rng(seed=seed)

    # generate potentially many k = 1, ..., K affine subspaces W of different dimension d << dim_n
    # - K = num_W_in_dataset
    # dim W should be at least 2? so dim_n >= 3?
    # dim W should be at most a fraction of the context_len --OR-- dim_n - 1
    if isinstance(style_subspace_dimensions, (np.integer, int)):
        assert 1 <= style_subspace_dimensions <= min(dim_n,  context_len//2)
        dim_d_k = style_subspace_dimensions * np.ones(num_W_in_dataset, dtype=int)  # all subspace same size
    else:
        assert style_subspace_dimensions == 'random'
        dim_d_k = np.random.randint(1, min(dim_n,  context_len//2), size=num_W_in_dataset)
    print('data_train_test_split_linear(...)')
    print('\tstyle_subspace_dimensions=%s' % style_subspace_dimensions)
    print('\tdim_d_k min/max:', dim_d_k.min(), dim_d_k.max())

    nsample_per_subspace = context_len  # alias | this is the length of the input sequence for sequence model
    assert context_len > max(dim_d_k)

    # sanity check
    train_plus_test_size = num_W_in_dataset * context_examples_per_W * samples_per_context_example
    print('data_train_test_split_linear(...)')
    print('\ttrain_plus_test_size (%d) = num_W_in_dataset (%d) x context_examples_per_W (%d) x samples_per_context_example (%d)' %
          (train_plus_test_size, num_W_in_dataset, context_examples_per_W, samples_per_context_example))

    # create X, y training blob
    x_total = []
    y_total = []
    data_subspace_dict = {j: {} for j in range(train_plus_test_size)}

    j = 0
    # for each dimensionality in the list above, generate a random m, V pair and use them to sample > d_max points
    for k, dim_d in enumerate(dim_d_k):

        # select offsets
        rand_m = gen_uniform_data(dim_n, rng=rng)
        if style_origin_subspace:
            rand_m = np.zeros(dim_n)

        # select basis (orthonormal via QR decomp)
        #rand_V = gen_uniform_data((dim_n, dim_d + 1))              # d+1 is used below to get extra orthog direction
        rand_V = gen_uniform_data((dim_n, dim_n), rng=rng)  # dims d+1 to n are used below to create extra orthog directions
        rand_V, _ = np.linalg.qr(rand_V)

        # to corrupt x_query (final vector in each training sequence), we will give it a kick in orthogonal direction
        W_random_orthog_direction = rand_V[:, dim_d:]  # we leverage this in the style_corruption_orthog = True case
        rand_V =                    rand_V[:, :dim_d]

        """ 
        - (A) mode where the corruption of the last token x_L is just a gaussian centered at x_L (i.e. not necc. orthogonal to the subspace)
        - (B) alt, the corruption is orthogonal to the subspace (using the last direction of the synthetic dim_d + 1 size basis)
        - consider two separate dataset modes: (A) and (B), could also interpolate between them as 0 <= alpha <= 1 and summing a A + (1-a) B
        """
        if style_corruption_orthog:
            # samples the corruption kicks (in orthogonal directions)
            '''
            # ORIGINAL WAY
            corruption_kicks = sample_from_subspace(0 * rand_m, W_random_orthog_direction, examples_per_W, seed=seed1)
            kick_mu, kick_sigma = sigma_corruption, 1.0  # was 1.0, 1.0 orig
            np.random.seed(seed1)
            kick_size = np.random.normal(loc=kick_mu, scale=kick_sigma, size=examples_per_W)
            corruption_kicks = corruption_kicks * kick_size  # rescales corruption kicks... need to clean this up...
            #corruption_kicks = corruption_kicks * 1  # rescales corruption kicks... need to clean this up...
            '''
            # TODO cleanup this block so that it matches the one below and lengths scale with sigma_corruption
            dim_W_perp = W_random_orthog_direction.shape[1]
            assert dim_W_perp == dim_n - dim_d
            corruption_cov = sigma2_corruption * np.eye(dim_W_perp)  # TODO replace by A @ A.T, given full rank A?
            c = gen_gaussian_data(
                np.zeros(dim_W_perp),
                corruption_cov,
                (context_examples_per_W), rng=rng)  # random coefficients for the basis {v_i}; shape samples x dim_d
            corruption_kicks = W_random_orthog_direction @ c.T
        else:
            corruption_mean = np.zeros(dim_n)
            corruption_cov = sigma2_corruption * np.eye(dim_n)

            # Alternative way of having non-isotropic corruption covariance
            """
            if corr_scaling_matrix is None:
                corruption_cov = sigma2_corruption * np.eye(dim_n)
            else:
                print('WARNING - corr_scaling_matrix is not none, setting and normalizing induced covariance...')
                # we normalize to original case by forcing trace to be sigma2_corruption * n
                # - since total cov in isotropic case is sigma2_corruption * n
                # - given any full rank A, we have the following steps:
                #   1) compute frobenius norm of A
                #   2) scale A as  A' = c A  with  c = sqrt(n) / ||A||_F
                #   3) compute sigma_arr = A' @ A'.T
                # - observe that Tr(sigma_arr) = Tr(A' @ A'.T) = ||A'||_F^2 = n   - matching identity matrix
                '''
                frob_norm_val = np.linalg.norm(corr_scaling_matrix, 'fro')  # square to get Tr(A @ A.T)
                corr_scaling_matrix_normed = (np.sqrt(dim_n) / frob_norm_val) * corr_scaling_matrix
                sigma_matrix_normed = corr_scaling_matrix_normed @ corr_scaling_matrix_normed.T
                print(np.linalg.trace(sigma_matrix_normed))
                print('='*20)
                '''
                corruption_cov = sigma2_corruption * sigma_matrix_normed  # TODO new form
            """
            corruption_kicks = gen_gaussian_data(corruption_mean, corruption_cov, context_examples_per_W, rng=rng)  # shape samples x dim_n
            corruption_kicks = corruption_kicks.T  # shape dim_n x context_examples_per_W

        for sample_idx in range(context_examples_per_W):
            # generate samples from the random subspace
            X_sequence = sample_from_subspace(rand_m, rand_V, nsample_per_subspace,
                                              sigma_Z_sqr=sigma2_pure_context, rng=rng, uniform=False)
            corruption_kicks_for_sample = corruption_kicks[:, sample_idx]

            # if non-isotropic covariance modification is used, we need to apply the scaling matrix
            if corr_scaling_matrix is not None:
                '''
                print('WARNING - corr_scaling_matrix is not none, setting and normalizing induced covariance...')
                frob_norm_val = np.linalg.norm(corr_scaling_matrix, 'fro')  # square to get Tr(A @ A.T)
                corr_scaling_matrix_normed = (np.sqrt(dim_n) / frob_norm_val) * corr_scaling_matrix'''
                print('WARNING - corr_scaling_matrix is not none...')
                X_sequence                  = corr_scaling_matrix @ X_sequence
                corruption_kicks_for_sample = corr_scaling_matrix @ corruption_kicks[:, sample_idx]

            # corruption of last column
            y_target = np.copy(X_sequence[:, -1])
            X_sequence[:, -1] = y_target + corruption_kicks_for_sample

            x_total.append(X_sequence)
            y_total.append(y_target)

            data_subspace_dict[j]['W_m'] = rand_m
            data_subspace_dict[j]['W_V'] = rand_V
            data_subspace_dict[j]['W_dim'] = dim_d
            j += 1

    x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_util(
        x_total, y_total, data_subspace_dict, context_len, dim_n, num_W_in_dataset,
        context_examples_per_W, test_ratio, as_torch=as_torch,
        savez_fname=savez_fname, verbose=verbose, rng=rng
    )

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


class DatasetWrapper(Dataset):
    """
    (relic): currently, there is a "remainder" batch at the end, with size smaller than batch_size -- could discard it
    """
    def __init__(self, X, Y):
        self.x = X
        self.y = Y

        print("what is self.x.size() in DatasetWrapper as X.shape and device and type", self.x.shape, self.x.device, self.x.dtype)
        self.dim_n = self.x.size()[1]
        self.context_length = self.x.size()[2]

    # Mandatory: Get input pair for training
    def __getitem__(self, idx):
        return self.x[idx, :, :], self.y[idx, :]

    # Mandatory: Number of elements in dataset (i.e. size of batch dimension 0)
    def __len__(self):
        X_len = self.x.size()[0]
        return X_len

    # This function is not needed
    def plot(self):
        print('not implemented')
        return
       
datagen_seed = 0
idx = 0
datagen_choice = 0
context_len = 500 # this is L
dim_n = 16 #this is the ambient dim I think n
test_ratio = 0.2

#datagen_seed = 15  # None  |  15, 4

###datagen_choice = 1   # {0, 1, 2} -> {linear, clusters, manifold}
datagen_label = ['linear'][datagen_choice]

sigma2_corruption = 0.1


base_kwargs = dict(
    context_len=context_len,
    dim_n=dim_n,
    num_W_in_dataset=1000,
    context_examples_per_W=1,
    samples_per_context_example=1,
    test_ratio=test_ratio,
    verbose=True,  
    as_torch=True,  
    savez_fname=None,  
    seed=datagen_seed,  
    style_subspace_dimensions=DATAGEN_GLOBALS[datagen_choice]['style_subspace_dimensions'],
    style_origin_subspace=DATAGEN_GLOBALS[datagen_choice]['style_origin_subspace'],
    style_corruption_orthog=DATAGEN_GLOBALS[datagen_choice]['style_corruption_orthog'],
    sigma2_corruption=sigma2_corruption,
)




# Leave this as is
def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    # output = model(xs, ys)
    output = model(xs)
    print("in train step device of xs ys", xs.device, ys.device) #yeah mps so ok
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

#TODO: this is from ICL but not sure if I need it now - how they do seeds in denoise
def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training["learning_rate"])

    #this is from icl
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        

    batch_size = args.training["batch_size"]
    
    # pbar = tqdm(range(starting_step, args.training["train_steps"])) TODO: put this back later

    
    x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_linear(
        **base_kwargs,
        sigma2_pure_context=DATAGEN_GLOBALS[datagen_choice]['sigma2_pure_context'])
    

    print("in train denoiser, shape and device of x_train", x_train.shape, x_train.device)
    train_dataset = DatasetWrapper(x_train, y_train)
    test_dataset = DatasetWrapper(x_test, y_test)
    
    nwork = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork) #TODO@DR: is this correct to shuffle in test?

    
    # inspect network class
    params = list(model.parameters())
    print('\nNum of params matrices to train:', len(params))
    print('\tparams[0].size():', params[0].size())


    loss_func  = nn.MSELoss()

    full_loss_sample_interval = 4 

    epochs = args.training["epochs"]
    #TODO@DR rename params for consistency - will want to use yaml and wandb

     ################################################################################
    # prep loss curves (x, y arrays)
    ################################################################################
    nbatches_per_epoch = np.ceil(x_train.shape[0] / batch_size)
    curve_x_losstrain_batch = np.arange(1, epochs * nbatches_per_epoch + 1) / nbatches_per_epoch
    curve_x_losstrain_epochs_avg = np.arange(1, epochs + 1)  # average over batches in the epoch (fast but inaccurate estimate)
    curve_x_losstest_interval = np.arange(0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch)
    curve_x_losstrain_interval = np.arange(0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch)

    # monitor the train error on the full test set every k batches (could be more/less than once per epoch)
    train_full_mse_loss = report_dataset_loss(model, loss_func, train_loader, 'train')
    # monitor the test error on the full test set every k batches (could be more/less than once per epoch)
    test_full_mse_loss = report_dataset_loss(model, loss_func, test_loader, 'test')

    curve_y_losstrain_epochs_avg = []  # will append to this each epoch
    curve_y_losstrain_batch = []  # we begin tracking AFTER the first batch (could get loss nograd first batch here)
    curve_y_losstest_interval = [test_full_mse_loss]  # will append to this each full_loss_sample_interval batches
    curve_y_losstrain_interval = [train_full_mse_loss]  # will append to this each epoch

    ################################################################################
    # train loop
    ################################################################################

    count = 1  # batch counter
    period_save_weights = 1 # how often to save manually 

    io_dict = {} #TODO@this has some function; put in yaml

    for epoch in range(epochs):
         
        #  if epoch % period_save_weights == 0:
        #     model_path = io_dict['dir_checkpoints'] + os.sep + 'model_e%d' % epoch + '.pth'
        #     torch.save(model.state_dict(), model_path)
        running_loss_epoch = 0.0
        running_loss_mesoscale = 0.0
        running_batch_counter = 0
        print('\nepoch:', epoch)

        for i, data in enumerate(train_loader, 0):
            inputs, targets = data 
            print("inputs shape and device", inputs.shape, inputs.device) #(batch size, x token dimension, context len)
            print("targets shape and device", targets.shape, targets.device) #(batch size, n dim of last token)
        

        

            loss, output = train_step(model, inputs.to(device), targets.to(device), optimizer, loss_func)
            print("loss ", loss)
            curve_y_losstrain_batch.append(loss)

            # print statistics
            running_loss_epoch     += curve_y_losstrain_batch[-1]  # was [count]
            running_loss_mesoscale += curve_y_losstrain_batch[-1]  # was [count]

            

            count += 1  # count tracks number of batches which have been trained over (at this point)
            running_batch_counter += 1
        print('end epoch:', epoch, '====================')
        curve_y_losstrain_epochs_avg.append(running_loss_epoch / running_batch_counter)

    print('Finished Training')
    train_loss_end = report_dataset_loss(model, loss_func, train_loader, 'train')
    test_loss_end = report_dataset_loss(model, loss_func, test_loader, 'test')

    print('curve_x_losstrain_epochs_avg', curve_x_losstrain_epochs_avg)
    print('curve_y_losstrain_epochs_avg', curve_y_losstrain_epochs_avg, '\n')

    print('curve_x_losstrain_batch', curve_x_losstrain_batch)
    plt.plot(curve_y_losstrain_epochs_avg)
    plt.show() # I can see epoch loss decreasing

    #OK: next - make this nice and train the linear case (on MPS for now)

            # point_wise_tags = list(range(curriculum.n_points))
            # # point_wise_loss_func = task.get_metric()

            # # point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)
            # # point_wise_loss = point_wise_loss_func(output, y_train.to(device)).mean(dim=0)
            # point_wise_loss = loss_func(output, y_train.to(device)).mean(dim=0)

            # baseline_loss = (
            #     sum(
            #         max(curriculum.n_dims_truncated - ii, 0)
            #         for ii in range(curriculum.n_points)
            #     )
            #     / curriculum.n_points
            # )

            # if i % args.wandb["log_every_steps"] == 0 and not args.test_run:
            #     wandb.log(
            #         {
            #             "overall_loss": loss,
            #             "excess_loss": loss / baseline_loss,
            #             "pointwise/loss": dict(
            #                 zip(point_wise_tags, point_wise_loss.cpu().numpy())
            #             ),
            #             "n_points": curriculum.n_points,
            #             "n_dims": curriculum.n_dims_truncated,
            #         },
            #         step=i,
            #     )

            # curriculum.update()

            # pbar.set_description(f"loss {loss}")
            # if i % args.training["save_every_steps"] == 0 and not args.test_run:
            #     training_state = {
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "train_step": i,
            #     }
            #     torch.save(training_state, state_path)

            # if (
            #     args.training["keep_every_steps"] > 0
            #     and i % args.training["keep_every_steps"] == 0
            #     and not args.test_run
            #     and i > 0
            # ):
            #     torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training["curriculum"]
        curriculum_args.points["start"] = curriculum_args.points["end"]
        curriculum_args.dims["start"] = curriculum_args.dims["end"]
        args.training["train_steps"] = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb["project"],
            entity=args.wandb["entity"],
            config=args.__dict__,
            notes=args.wandb["notes"],
            name=args.wandb["name"],
            resume=True,
        )

    # model = build_model(args.model)
    # model = TransformerModelV1nores(context_len, dim_n)
    model = TransformerModelV2nores(context_len, dim_n)
    # model.cuda()
    model.to(device)
    model.train()

    train(model, args)

    # if not args.test_run:
    #     _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My application description.")
    # args = parser.parse_quinfig()
    parser.add_argument("--config-file", help="Path to YAML config file")
    parser.add_argument("--model", default="default_value")
    parser.add_argument("--test_run", default=None)

    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
            parser.set_defaults(**config)
        args = parser.parse_args() # Reload arguments to apply YAML values


    # assert args.model["family"] in ["gpt2", "lstm"]
    # print(f"Running with: {args}")

   
    # if not args.test_run:
    #     run_id = args.training["resume_id"]
    #     if run_id is None:
    #         run_id = str(uuid.uuid4())

    #     out_dir = os.path.join(args.out_dir, run_id)
    #     if not os.path.exists(out_dir):
    #         os.makedirs(out_dir)
    #     args.out_dir = out_dir

    #     with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
    #         yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)

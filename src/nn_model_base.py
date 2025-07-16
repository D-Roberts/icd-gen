import os
import sys

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

# from data_io import load_runinfo_from_rundir
# from data_tools import data_train_test_split_linear
# from settings import DIR_MODELS
# from torch_device import device_select

DIR_CURRENT = os.path.dirname(__file__)
DIR_PARENT = os.path.dirname(DIR_CURRENT)
sys.path.append(DIR_PARENT)

# default kwargs for dataset generation for each case 0: 'Linear', 1: 'Clustering', 2: 'Manifold'
DATAGEN_GLOBALS = {
    0: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,  # if True, the noise is only in orthogonal directions
        style_origin_subspace=True,  # if True, the subspace must contain origin (not affine)
        style_subspace_dimensions="random",  # int or 'random'
        # parameters specific to the Linear case
        sigma2_pure_context=2.0,  # controls radius of ball of pure of samples (default: 2.0)
        corr_scaling_matrix=None,  # x = Pz; y = A x  \tilde x = A (x + z); (normally identity I_n)
    ),
    1: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,  # mandatory for this case
        style_origin_subspace=True,  # mandatory for this case
        style_subspace_dimensions="full",  # full means dim-n gaussian balls in R^n
        # parameters specific to the Clustering case
        style_cluster_mus="unit_norm",
        style_cluster_vars="isotropic",
        num_cluster=3,
        cluster_var=0.01,  # controls radius of ball of pure of samples (default: 2.0); could be a list in Clustering case
    ),
    2: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,  # mandatory for this case
        style_origin_subspace=True,  # mandatory for this case
        style_subspace_dimensions="random",  # int or 'random'
        # parameters specific to the Manifold case
        radius_sphere=1.0,  # controls radius of sphere (d-dim manifold S in R^n)
    ),
}

# asserts for Linear case
assert DATAGEN_GLOBALS[0]["style_subspace_dimensions"] in ["random"] or isinstance(
    DATAGEN_GLOBALS[0]["style_subspace_dimensions"], int
)
assert DATAGEN_GLOBALS[0]["style_corruption_orthog"] is False
assert DATAGEN_GLOBALS[0]["style_origin_subspace"] is True


DIR_OUT = DIR_PARENT + os.sep + "output"
DIR_DATA = DIR_PARENT + os.sep + "input"
DIR_MODELS = DIR_PARENT + os.sep + "models"
DIR_RUNS = DIR_PARENT + os.sep + "runs"

for core_dir in [DIR_OUT, DIR_DATA, DIR_MODELS, DIR_RUNS]:
    if not os.path.exists(core_dir):
        os.mkdir(core_dir)


def load_runinfo_from_rundir(dir_run, model_prefix=""):
    """
    Load runinfo.txt from a given run directory
    """
    runinfo_fpath = dir_run + os.sep + model_prefix + "runinfo.txt"
    with open(runinfo_fpath, "r") as runinfo:
        runinfo_lines = runinfo.readlines()

    # step: convert runinfo to a dictionary and return it
    runinfo_dict = {}
    for line in runinfo_lines:
        if line.split(",")[0].strip() == "scheduler":
            key = line.split(",")[0]
            val = ",".join(line.split(",")[1:]).strip()
            val = eval(val)
            assert isinstance(val, dict)
        else:
            key, val = line.split(",")
            key, val = key.strip(), val.strip()
        runinfo_dict[key] = val

    # handle potentially ambiguous key -> value types
    # - style_subspace_dimensions (int or str)
    # - seed_dataset (None or int)
    # - could have diff number of keys relating to gradient descent e.g. adam_lr, sgd_lr, etc. -- keep as str
    for key in [
        "epochs",
        "batch_size",
        "dim_n",
        "context_len",
        "train_plus_test_size",
        "full_loss_sample_interval",
        "context_examples_per_W",
        "samples_per_context_example",
        "num_W_in_dataset",
    ]:
        runinfo_dict[key] = int(runinfo_dict[key])
    for key in ["style_subspace_dimensions"]:
        if runinfo_dict[key] != "full":
            runinfo_dict[key] = int(runinfo_dict[key])
    for key in ["sigma2_corruption", "sigma2_pure_context", "test_ratio"]:
        runinfo_dict[key] = float(runinfo_dict[key])
    for key in ["style_corruption_orthog", "style_origin_subspace"]:
        runinfo_dict[key] = (
            runinfo_dict[key] == "True"
        )  # if True, the bool val is True, else False

    # specific checks for particular datagen cases
    if "case1_specific_style_cluster_mus" in runinfo_dict.keys():
        runinfo_dict["style_cluster_mus"] = runinfo_dict[
            "case1_specific_style_cluster_mus"
        ]
        del runinfo_dict["case1_specific_style_cluster_mus"]

    if "case1_specific_style_cluster_vars" in runinfo_dict.keys():
        runinfo_dict["style_cluster_vars"] = runinfo_dict[
            "case1_specific_style_cluster_vars"
        ]
        del runinfo_dict["case1_specific_style_cluster_vars"]

    if "case1_specific_cluster_var" in runinfo_dict.keys():
        runinfo_dict["cluster_var"] = float(runinfo_dict["case1_specific_cluster_var"])
        del runinfo_dict["case1_specific_cluster_var"]

    if "case1_specific_num_cluster" in runinfo_dict.keys():
        runinfo_dict["num_cluster"] = int(runinfo_dict["case1_specific_num_cluster"])
        del runinfo_dict["case1_specific_num_cluster"]
    return runinfo_dict


# asserts for Linear case
assert DATAGEN_GLOBALS[0]["style_subspace_dimensions"] in ["random"] or isinstance(
    DATAGEN_GLOBALS[0]["style_subspace_dimensions"], int
)
assert DATAGEN_GLOBALS[0]["style_corruption_orthog"] is False
assert DATAGEN_GLOBALS[0]["style_origin_subspace"] is True


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
    arr = rng.multivariate_normal(
        mean, cov, size=size, check_valid="raise"
    )  # size x dim n
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
            (dim_W, nsample), rng=rng
        )  # random coefficients for the basis {v_i}
        # note by default, c will have a variance of 1/3 (U[-1,1]); we can scale it by
        c = (
            np.sqrt(3 * sigma_Z_sqr) * c
        )  # e.g. if goal is variance of 10, then scale sqrt(3 * 10)
        rand_samples = np.expand_dims(W_m, axis=1) + W_V @ c
    else:
        # gaussian ball method
        x_blob = rng.multivariate_normal(W_m, sigma_Z_sqr * np.eye(dim_n), nsample)
        projector_exact = W_V @ np.linalg.inv(W_V.T @ W_V) @ W_V.T
        projection_b_onto_W = np.expand_dims(W_m, axis=1) + projector_exact @ (
            x_blob.T - np.expand_dims(W_m, axis=1)
        )
        rand_samples = projection_b_onto_W
    return rand_samples


def data_train_test_split_util(
    x_total,
    y_total,
    data_subspace_dict,
    context_len,
    dim_n,
    num_W_in_dataset,
    context_examples_per_W,
    test_ratio,
    as_torch=True,
    savez_fname=None,
    verbose=True,
    rng=None,
):
    """
    Used commonly by data_train_test_split_* functions - for * = {linear, clusters, manifold}

    TODO make it so test ratio 1.0 or 0.0 makes the corresponding array empty (as opposed to nasty if/else currently)
    """
    if verbose:
        print(
            "Generating train/test data for NN training (context length = %d)..."
            % context_len
        )
        print(
            "\tcontext_len=%d, dim_n=%d, num_W_in_dataset=%d, examples_per_W=%d, test_ratio=%s"
            % (context_len, dim_n, num_W_in_dataset, context_examples_per_W, test_ratio)
        )
        print("\tnsample_per_subspace (context_len):", context_len)
        print("Total dataset size before split: %d (x,y) pairs" % len(y_total))

    x_total = np.array(x_total).astype(np.float32)
    y_total = np.array(y_total).astype(np.float32)

    rng = (
        rng or np.random.default_rng()
    )  # determines how dataset is split; if no rng passed, create one

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
            print("\t x_train:", x_train.shape)
            print("\t y_train:", y_train.shape)
            print("\t x_test: None")
            print("\t y_test: None")

        if savez_fname is not None:
            assert savez_fname[-4:] == ".npz"
            # save dataset to file
            dataset_fpath = savez_fname
            print("\nSaving dataset to file...", dataset_fpath)
            np.savez_compressed(
                dataset_fpath,
                x_train=x_train,
                y_train=y_train,
                x_test=np.empty(1),
                y_test=np.empty(1),
            )

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
            print("\t x_train:", x_train.shape)
            print("\t y_train:", y_train.shape)
            print("\t x_test:", x_test.shape)
            print("\t y_test:", y_test.shape)

        if savez_fname is not None:
            assert savez_fname[-4:] == ".npz"
            # save dataset to file
            dataset_fpath = savez_fname
            print("\nSaving dataset to file...", dataset_fpath)
            np.savez_compressed(
                dataset_fpath,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

        if as_torch:
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
            x_test = torch.from_numpy(x_test)
            y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


def data_train_test_split_linear(
    context_len=100,
    dim_n=8,  # was 128
    num_W_in_dataset=100,
    context_examples_per_W=1,
    samples_per_context_example=1,
    test_ratio=0.2,
    verbose=True,
    seed=None,
    style_subspace_dimensions=DATAGEN_GLOBALS[0]["style_subspace_dimensions"],
    style_origin_subspace=DATAGEN_GLOBALS[0][
        "style_origin_subspace"
    ],  # TODO convert to float  0 < m < 1 magnitude
    style_corruption_orthog=DATAGEN_GLOBALS[0][
        "style_corruption_orthog"
    ],  # TODO convert to float  0 < a < 1
    sigma2_corruption=DATAGEN_GLOBALS[0]["sigma2_corruption"],
    sigma2_pure_context=DATAGEN_GLOBALS[0]["sigma2_pure_context"],
    corr_scaling_matrix=DATAGEN_GLOBALS[0]["corr_scaling_matrix"],
    savez_fname=None,
    as_torch=True,
):
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
    print("data_train_test_split_linear() args...")
    print("\tstyle_subspace_dimensions:", style_subspace_dimensions)
    print("\tstyle_origin_subspace:", style_origin_subspace)
    print("\tstyle_corruption_orthog:", style_corruption_orthog)
    print("\tcontext_len:", context_len)
    print("\tdim_n:", dim_n)
    print("\tnum_W_in_dataset:", num_W_in_dataset)
    print("\tcontext_examples_per_W:", context_examples_per_W)
    print("\tsamples_per_context_example:", samples_per_context_example)
    print("\tsigma_corruption:", sigma2_corruption)
    print("\tsigma_pure_context:", sigma2_pure_context)
    print("\tcorr_scaling_matrix:", corr_scaling_matrix)
    print("\tseed:", seed)
    print("\tsavez_fname:", savez_fname)

    assert samples_per_context_example == 1

    rng = np.random.default_rng(seed=seed)

    # generate potentially many k = 1, ..., K affine subspaces W of different dimension d << dim_n
    # - K = num_W_in_dataset
    # dim W should be at least 2? so dim_n >= 3?
    # dim W should be at most a fraction of the context_len --OR-- dim_n - 1
    if isinstance(style_subspace_dimensions, (np.integer, int)):
        assert 1 <= style_subspace_dimensions <= min(dim_n, context_len // 2)
        dim_d_k = style_subspace_dimensions * np.ones(
            num_W_in_dataset, dtype=int
        )  # all subspace same size
    else:
        assert style_subspace_dimensions == "random"
        dim_d_k = np.random.randint(
            1, min(dim_n, context_len // 2), size=num_W_in_dataset
        )
    print("data_train_test_split_linear(...)")
    print("\tstyle_subspace_dimensions=%s" % style_subspace_dimensions)
    print("\tdim_d_k min/max:", dim_d_k.min(), dim_d_k.max())

    nsample_per_subspace = context_len  # alias | this is the length of the input sequence for sequence model
    assert context_len > max(dim_d_k)

    # sanity check
    train_plus_test_size = (
        num_W_in_dataset * context_examples_per_W * samples_per_context_example
    )
    print("data_train_test_split_linear(...)")
    print(
        "\ttrain_plus_test_size (%d) = num_W_in_dataset (%d) x context_examples_per_W (%d) x samples_per_context_example (%d)"
        % (
            train_plus_test_size,
            num_W_in_dataset,
            context_examples_per_W,
            samples_per_context_example,
        )
    )

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
        # rand_V = gen_uniform_data((dim_n, dim_d + 1))              # d+1 is used below to get extra orthog direction
        rand_V = gen_uniform_data(
            (dim_n, dim_n), rng=rng
        )  # dims d+1 to n are used below to create extra orthog directions
        rand_V, _ = np.linalg.qr(rand_V)

        # to corrupt x_query (final vector in each training sequence), we will give it a kick in orthogonal direction
        W_random_orthog_direction = rand_V[
            :, dim_d:
        ]  # we leverage this in the style_corruption_orthog = True case
        rand_V = rand_V[:, :dim_d]

        """ 
        - (A) mode where the corruption of the last token x_L is just a gaussian centered at x_L (i.e. not necc. orthogonal to the subspace)
        - (B) alt, the corruption is orthogonal to the subspace (using the last direction of the synthetic dim_d + 1 size basis)
        - consider two separate dataset modes: (A) and (B), could also interpolate between them as 0 <= alpha <= 1 and summing a A + (1-a) B
        """
        if style_corruption_orthog:
            # samples the corruption kicks (in orthogonal directions)
            """
            # ORIGINAL WAY
            corruption_kicks = sample_from_subspace(0 * rand_m, W_random_orthog_direction, examples_per_W, seed=seed1)
            kick_mu, kick_sigma = sigma_corruption, 1.0  # was 1.0, 1.0 orig
            np.random.seed(seed1)
            kick_size = np.random.normal(loc=kick_mu, scale=kick_sigma, size=examples_per_W)
            corruption_kicks = corruption_kicks * kick_size  # rescales corruption kicks... need to clean this up...
            #corruption_kicks = corruption_kicks * 1  # rescales corruption kicks... need to clean this up...
            """
            # TODO cleanup this block so that it matches the one below and lengths scale with sigma_corruption
            dim_W_perp = W_random_orthog_direction.shape[1]
            assert dim_W_perp == dim_n - dim_d
            corruption_cov = sigma2_corruption * np.eye(
                dim_W_perp
            )  # TODO replace by A @ A.T, given full rank A?
            c = gen_gaussian_data(
                np.zeros(dim_W_perp), corruption_cov, (context_examples_per_W), rng=rng
            )  # random coefficients for the basis {v_i}; shape samples x dim_d
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
            corruption_kicks = gen_gaussian_data(
                corruption_mean, corruption_cov, context_examples_per_W, rng=rng
            )  # shape samples x dim_n
            corruption_kicks = (
                corruption_kicks.T
            )  # shape dim_n x context_examples_per_W

        for sample_idx in range(context_examples_per_W):
            # generate samples from the random subspace
            X_sequence = sample_from_subspace(
                rand_m,
                rand_V,
                nsample_per_subspace,
                sigma_Z_sqr=sigma2_pure_context,
                rng=rng,
                uniform=False,
            )
            corruption_kicks_for_sample = corruption_kicks[:, sample_idx]

            # if non-isotropic covariance modification is used, we need to apply the scaling matrix
            if corr_scaling_matrix is not None:
                """
                print('WARNING - corr_scaling_matrix is not none, setting and normalizing induced covariance...')
                frob_norm_val = np.linalg.norm(corr_scaling_matrix, 'fro')  # square to get Tr(A @ A.T)
                corr_scaling_matrix_normed = (np.sqrt(dim_n) / frob_norm_val) * corr_scaling_matrix
                """
                print("WARNING - corr_scaling_matrix is not none...")
                X_sequence = corr_scaling_matrix @ X_sequence
                corruption_kicks_for_sample = (
                    corr_scaling_matrix @ corruption_kicks[:, sample_idx]
                )

            # corruption of last column
            y_target = np.copy(X_sequence[:, -1])
            X_sequence[:, -1] = y_target + corruption_kicks_for_sample

            x_total.append(X_sequence)
            y_total.append(y_target)

            data_subspace_dict[j]["W_m"] = rand_m
            data_subspace_dict[j]["W_V"] = rand_V
            data_subspace_dict[j]["W_dim"] = dim_d
            j += 1

    x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = (
        data_train_test_split_util(
            x_total,
            y_total,
            data_subspace_dict,
            context_len,
            dim_n,
            num_W_in_dataset,
            context_examples_per_W,
            test_ratio,
            as_torch=as_torch,
            savez_fname=savez_fname,
            verbose=verbose,
            rng=rng,
        )
    )

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


# gen data for linear only, skip vis
class DatasetWrapper(Dataset):
    """
    (relic): currently, there is a "remainder" batch at the end, with size smaller than batch_size -- could discard it
    """

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

        print("what is self.x.size()", self.x.size())
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
        print("not implemented")
        return


datagen_seed = 0
idx = 0
datagen_choice = 0
context_len = 30  # this is L
dim_n = 8  # this is the ambient dim I think n
test_ratio = 0.2

# datagen_seed = 15  # None  |  15, 4

###datagen_choice = 1   # {0, 1, 2} -> {linear, clusters, manifold}
datagen_label = ["linear"][datagen_choice]

sigma2_corruption = 0.1

DIR_CURRENT = os.path.dirname(__file__)
DIR_PARENT = os.path.dirname(DIR_CURRENT)
sys.path.append(DIR_PARENT)

DIR_OUT = DIR_PARENT + os.sep + "output"

# default kwargs for dataset generation for each case 0: 'Linear', 1: 'Clustering', 2: 'Manifold'
DATAGEN_GLOBALS = {
    0: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,  # if True, the noise is only in orthogonal directions
        style_origin_subspace=True,  # if True, the subspace must contain origin (not affine)
        style_subspace_dimensions="random",  # int or 'random'
        # parameters specific to the Linear case
        sigma2_pure_context=2.0,  # controls radius of ball of pure of samples (default: 2.0)
        corr_scaling_matrix=None,  # x = Pz; y = A x  \tilde x = A (x + z); (normally identity I_n)
    ),
    1: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,  # mandatory for this case
        style_origin_subspace=True,  # mandatory for this case
        style_subspace_dimensions="full",  # full means dim-n gaussian balls in R^n
        # parameters specific to the Clustering case
        style_cluster_mus="unit_norm",
        style_cluster_vars="isotropic",
        num_cluster=3,
        cluster_var=0.01,  # controls radius of ball of pure of samples (default: 2.0); could be a list in Clustering case
    ),
    2: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,  # mandatory for this case
        style_origin_subspace=True,  # mandatory for this case
        style_subspace_dimensions="random",  # int or 'random'
        # parameters specific to the Manifold case
        radius_sphere=1.0,  # controls radius of sphere (d-dim manifold S in R^n)
    ),
}

# asserts for Linear case
assert DATAGEN_GLOBALS[0]["style_subspace_dimensions"] in ["random"] or isinstance(
    DATAGEN_GLOBALS[0]["style_subspace_dimensions"], int
)
assert DATAGEN_GLOBALS[0]["style_corruption_orthog"] is False
assert DATAGEN_GLOBALS[0]["style_origin_subspace"] is True


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
    arr = rng.multivariate_normal(
        mean, cov, size=size, check_valid="raise"
    )  # size x dim n
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
            (dim_W, nsample), rng=rng
        )  # random coefficients for the basis {v_i}
        # note by default, c will have a variance of 1/3 (U[-1,1]); we can scale it by
        c = (
            np.sqrt(3 * sigma_Z_sqr) * c
        )  # e.g. if goal is variance of 10, then scale sqrt(3 * 10)
        rand_samples = np.expand_dims(W_m, axis=1) + W_V @ c
    else:
        # gaussian ball method
        x_blob = rng.multivariate_normal(W_m, sigma_Z_sqr * np.eye(dim_n), nsample)
        projector_exact = W_V @ np.linalg.inv(W_V.T @ W_V) @ W_V.T
        projection_b_onto_W = np.expand_dims(W_m, axis=1) + projector_exact @ (
            x_blob.T - np.expand_dims(W_m, axis=1)
        )
        rand_samples = projection_b_onto_W
    return rand_samples


def data_train_test_split_util(
    x_total,
    y_total,
    data_subspace_dict,
    context_len,
    dim_n,
    num_W_in_dataset,
    context_examples_per_W,
    test_ratio,
    as_torch=True,
    savez_fname=None,
    verbose=True,
    rng=None,
):
    """
    Used commonly by data_train_test_split_* functions - for * = {linear, clusters, manifold}

    TODO make it so test ratio 1.0 or 0.0 makes the corresponding array empty (as opposed to nasty if/else currently)
    """
    if verbose:
        print(
            "Generating train/test data for NN training (context length = %d)..."
            % context_len
        )
        print(
            "\tcontext_len=%d, dim_n=%d, num_W_in_dataset=%d, examples_per_W=%d, test_ratio=%s"
            % (context_len, dim_n, num_W_in_dataset, context_examples_per_W, test_ratio)
        )
        print("\tnsample_per_subspace (context_len):", context_len)
        print("Total dataset size before split: %d (x,y) pairs" % len(y_total))

    x_total = np.array(x_total).astype(np.float32)
    y_total = np.array(y_total).astype(np.float32)

    rng = (
        rng or np.random.default_rng()
    )  # determines how dataset is split; if no rng passed, create one

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
            print("\t x_train:", x_train.shape)
            print("\t y_train:", y_train.shape)
            print("\t x_test: None")
            print("\t y_test: None")

        if savez_fname is not None:
            assert savez_fname[-4:] == ".npz"
            # save dataset to file
            dataset_fpath = savez_fname
            print("\nSaving dataset to file...", dataset_fpath)
            np.savez_compressed(
                dataset_fpath,
                x_train=x_train,
                y_train=y_train,
                x_test=np.empty(1),
                y_test=np.empty(1),
            )

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
            print("\t x_train:", x_train.shape)
            print("\t y_train:", y_train.shape)
            print("\t x_test:", x_test.shape)
            print("\t y_test:", y_test.shape)

        if savez_fname is not None:
            assert savez_fname[-4:] == ".npz"
            # save dataset to file
            dataset_fpath = savez_fname
            print("\nSaving dataset to file...", dataset_fpath)
            np.savez_compressed(
                dataset_fpath,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )

        if as_torch:
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
            x_test = torch.from_numpy(x_test)
            y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


base_kwargs = dict(
    context_len=context_len,
    dim_n=dim_n,
    num_W_in_dataset=64,
    context_examples_per_W=1,
    samples_per_context_example=1,
    test_ratio=test_ratio,
    verbose=True,
    as_torch=False,  # it was false and numpies outputed
    savez_fname=None,
    seed=datagen_seed,
    style_subspace_dimensions=DATAGEN_GLOBALS[datagen_choice][
        "style_subspace_dimensions"
    ],
    style_origin_subspace=DATAGEN_GLOBALS[datagen_choice]["style_origin_subspace"],
    style_corruption_orthog=DATAGEN_GLOBALS[datagen_choice]["style_corruption_orthog"],
    sigma2_corruption=sigma2_corruption,
)

# test out training the icl models directly for icl denoise linear task


# AVAIL_GPUS = min(1, torch.cuda.device_count())
# if not torch.backends.mps.is_available():
#     print('\nMPS device not found.')
#     mps_device = None

# if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         mps_device = torch.device("mps")
#         x = torch.ones(1, device=device)
#         print('\nCheck M1 chip:', x)
# elif torch.cuda.is_available():
#         device = torch.device("cuda:0")
# else:
#         device = "cpu"
# print('device selected:', device)

device = torch.device("mps")


class ReturnLastToken(nn.Module):
    """
    Baseline model -- return final token
    """

    def __init__(self):
        super().__init__()

    def forward(self, xs):
        outs = xs[:, :, -1]  # return the last token
        return outs


def weight_matrix(dim_in, dim_out, mode="default"):
    """
    Can use to initialize weight matrices in nn layers
        e.g. self.W_v = weight_matrix(h=ndim, w=ndim, mode="default")

    Throughout, we multiply on the right (e.g. y = W @ x) for consistency with the math notation.
        Thus, dim_in is the number of columns, and dim_out is the number of rows. (i.e. w, h in PyTorch notation)

    For info on default init from torch method nn.Linear, see here:
      https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    W_tensor = torch.empty(dim_out, dim_in)
    if mode == "default":
        low = -1.0 / np.sqrt(dim_in)
        high = 1.0 / np.sqrt(dim_in)
        torch.nn.init.uniform_(W_tensor, a=low, b=high)
    elif mode == "kaiming":
        torch.nn.init.kaiming_uniform_(W_tensor)
    elif mode == "normal":
        torch.nn.init.normal_(W_tensor, mean=0, std=0.02)
    else:
        raise ValueError("Unsupported `mode`")
    return torch.nn.Parameter(W_tensor)


class TransformerModelV1(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - `linear self-attention` (no softmax wrapper used)

    Notes
     - dim_input - the dimension of input tokens
     - dim_attn  - the dimension of the residual stream (attention head + MLP input and output)
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = context_length  # scaling used in Bartlett 2023

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens
        print("the device of xs on line 829****** within the V1 model archi", xs.device)

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = xs + W_PV @ xs @ attn_arg

        print("the shape and the device of f_attn", f_attn.shape, f_attn.device)

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        print("the shape and device of out in the base model", out.shape, out.device)

        return out


class TransformerModelV1nores(TransformerModelV1):
    """
    See docstring TransformerModelV1
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """

        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = (
            W_PV @ xs @ attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV1noresForceDiag(nn.Module):
    """
    See docstring TransformerModelV1
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        # self.W_KQ = weight_matrix(dim_input, dim_input, mode='normal')
        # self.W_PV = weight_matrix(dim_input, dim_input, mode='normal')

        self.W_KQ = torch.nn.Parameter(torch.tensor(0.1))
        self.W_PV = torch.nn.Parameter(torch.tensor(0.1))

        # self.W_KQ = torch.nn.Parameter(0.1 * torch.eye(dim_input))
        # self.W_PV = torch.nn.Parameter(0.1 * torch.eye(dim_input))
        self.rho = context_length  # scaling used in Bartlett 2023

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ * torch.eye(
            n_dim
        )  # self.W_KQ is a 1-parameter scalar --> make n x n diag arr
        W_PV = self.W_PV * torch.eye(
            n_dim
        )  # self.W_PV is a 1-parameter scalar --> make n x n diag arr

        rho = n_tokens

        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / rho
        f_attn = (
            W_PV @ xs @ attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV1noresOmitLast(TransformerModelV1):
    """
    See docstring TransformerModelV1
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs[:, :, [-1]]
        out = f_attn_approx[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV1noresForceDiagAndOmitLast(nn.Module):
    """
    See docstring TransformerModelV1
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_KQ = torch.nn.Parameter(torch.tensor(0.1))
        self.W_PV = torch.nn.Parameter(torch.tensor(0.1))
        self.rho = context_length

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ * torch.eye(
            n_dim
        )  # self.W_KQ is a 1-parameter scalar --> make n x n diag arr
        W_PV = self.W_PV * torch.eye(
            n_dim
        )  # self.W_PV is a 1-parameter scalar --> make n x n diag arr

        rho = n_tokens - 1

        xs_skip_last = xs[:, :, :-1]

        # attn_arg = torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs_skip_last / rho
        projection_estimate = xs_skip_last @ torch.transpose(xs_skip_last, 1, 2) / rho

        f_attn_approx = W_PV @ projection_estimate @ W_KQ @ xs
        out = f_attn_approx[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelV2(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1  # TODO implement...
        assert n_head == 1  # TODO implement...
        assert (
            dim_attn is None
        )  # TODO implement... for now we take dim_attn == dim_input
        # TODO in multilayer version, add AttnHead class beneath AttnLayer class? forward pass is just loop over nlayer

        # attention matrices (need to split by head...)
        self.W_KQ = weight_matrix(dim_input, dim_input, mode="default")
        self.W_PV = weight_matrix(dim_input, dim_input, mode="default")
        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # new line: now scaling is a fixed constant as in original QKV-attention - 1/sqrt(n)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs / self.rho
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = xs + W_PV @ xs @ softmax_attn_arg

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV2nores(TransformerModelV2):
    """
    See docstring TransformerModelV2
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV

        # faster to just use final token as the query, not whole context (we throw it away later)
        attn_arg = torch.transpose(xs, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho

        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ xs @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches
        return out


class TransformerModelV2noresOmitLast(TransformerModelV2):
    """
    See docstring TransformerModelV2
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__(
            context_length, dim_input, dim_attn=dim_attn, n_layer=n_layer, n_head=n_head
        )

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        batchsz, n_dim, n_tokens = xs.size()

        W_KQ = self.W_KQ
        W_PV = self.W_PV
        # rho = n_tokens

        xs_skip_last = xs[:, :, :-1]
        attn_arg = (
            torch.transpose(xs_skip_last, 1, 2) @ W_KQ @ xs[:, :, [-1]] / self.rho
        )

        # p7 Bartlett: "Softmax applied column-wise" (dim = data dim, not token dim)
        softmax_attn_arg = torch.softmax(attn_arg, dim=1)
        f_attn = (
            W_PV @ xs_skip_last @ softmax_attn_arg
        )  # the residual stream term "+ xs" has been removed

        out = f_attn[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


class TransformerModelQKVnores(nn.Module):
    """
    Simplest model:
    - no positional encoding is used
    - same as V1 but now softmax in place of `linear` self-attention
    """

    def __init__(self, context_length, dim_input, dim_attn=None, n_layer=1, n_head=1):
        super().__init__()
        assert n_layer == 1
        assert n_head == 1
        assert dim_attn is None

        # attention matrices (need to split by head...)
        self.W_Q = weight_matrix(dim_input, dim_input, mode="default")
        self.W_K = weight_matrix(dim_input, dim_input, mode="default")
        self.W_V = weight_matrix(dim_input, dim_input, mode="default")

        self.rho = 1.0

    def forward(self, xs):
        """
        xs is a sequence array of shape [batchsz, ndim, context_length]
            - batchsz = batch size
            - note the last two components match the math notation
        """
        ####batchsz, n_dim, n_tokens = xs.size()

        Q = self.W_Q @ xs[:, :, [-1]]
        K = self.W_K @ xs
        V = self.W_V @ xs

        # QK_d = (Q @ K.T) / self.rho
        KQ_d = (
            torch.transpose(K, 1, 2) @ Q / self.rho
        )  # this is tensor-argument of softmax attention
        prob = torch.softmax(KQ_d, dim=1)
        attention = V @ prob

        out = attention[
            :, :, -1
        ]  # take dim_n output result at last token, for all batches

        return out


def load_model_from_rundir(dir_run, epoch_int=None):
    """
    Step 0: assume rundir has structure used in io_dict above
    Step 1: read dim_n and context length from runinfo.txt
    Step 2: load model at particular epoch from model_checkpoints (if epoch unspecified, load model_end.pth)
    """
    # load runinfo settings
    runinfo_dict = load_runinfo_from_rundir(dir_run)
    dim_n = runinfo_dict["dim_n"]
    context_length = runinfo_dict["context_len"]
    nn_model_str = runinfo_dict["model"]
    epochs = runinfo_dict["epochs"]

    if epoch_int is not None:
        model_fpath = (
            dir_run
            + os.sep
            + "model_checkpoints"
            + os.sep
            + "model_e%d.pth" % epoch_int
        )
    else:
        model_fpath = (
            dir_run + os.sep + "model_checkpoints" + os.sep + "model_final.pth"
        )

    # nn_model_str is a string like 'TransformerModelV1' or its alias 'TV1' (short form)
    if nn_model_str in MODEL_CLASS_ALIAS_TO_STR.keys():
        nn_model_str = MODEL_CLASS_ALIAS_TO_STR[nn_model_str]
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]["class"]
    else:
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]["class"]

    net = nn_class(
        context_length, dim_n
    )  # TODO this currently assumes all models have same two inputs
    print("loading model at:", model_fpath, "...")
    net.load_state_dict(torch.load(model_fpath))
    net.eval()

    return net


def load_modeltype_from_fname(fname, dir_models=DIR_MODELS, model_params=None):
    """
    Trained model checkpoint files are assumed to follow a certain naming convention, and are placed in DIR_MODELS
        e.g. models\basicnetmult_chkpt_e240_L100_n128.pth

    If model_params is None, then the model params will be inferred from the filename itself
    """
    print("\nLoading model checkpoint from file...")
    fpath = dir_models + os.sep + fname
    print("...", fpath)

    if model_params is None:
        print("model_params is None; inferring class init from filename...")
        model_type = fname.split("_")[0]
        context_length = int(fname.split("_L")[1].split("_")[0])
        dim_input = int(fname.split("_n")[1].split("_")[0])
    else:
        model_type = model_params["nn_model"]
        context_length = int(model_params["context_length"])
        dim_input = int(model_params["dim_input"])

    # nn_model_str is a string like 'TransformerModelV1' or its alias 'TV1' (short form)
    if model_type in MODEL_CLASS_ALIAS_TO_STR.keys():
        nn_model_str = MODEL_CLASS_ALIAS_TO_STR[model_type]
        nn_class = MODEL_CLASS_FROM_STR[nn_model_str]["class"]
    else:
        nn_class = MODEL_CLASS_FROM_STR[model_type]["class"]

    print(
        "class:",
        model_type,
        "\n\tcontext_length=%d, dim_input=%d" % (context_length, dim_input),
    )
    net = nn_class(context_length, dim_input)
    print("loading weights from fpath:", fpath)
    net.load_state_dict(
        torch.load(fpath, weights_only=True)
    )  # avoid FutureWarning in latest Torch ~2.5

    return net, model_type, context_length, dim_input


def count_parameters(model):
    """
    Use: print parameters of torch nn.Module in nice manner
    From: https://stackoverflow.com/questions/67546610/pretty-print-list-without-module-in-an-ascii-table

    Table is just a list of lists
    """

    def pretty_print(table, ch1="-", ch2="|", ch3="+"):
        if len(table) == 0:
            return
        max_lengths = [
            max(column)
            for column in zip(*[[len(cell) for cell in row] for row in table])
        ]
        for row in table:
            print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))
            print(
                ch2.join(
                    [
                        "",
                        *[
                            ("{:<" + str(l) + "}").format(c)
                            for l, c in zip(max_lengths, row)
                        ],
                        "",
                    ]
                )
            )
        print(ch3.join(["", *[ch1 * l for l in max_lengths], ""]))

    total_params = 0
    table = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.append([name, str(params)])
        total_params += params
    print("(Modules | Parameters)")
    pretty_print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


MODEL_CLASS_FROM_STR = {
    "TransformerModelV1": {"class": TransformerModelV1, "alias": "TV1"},
    "TransformerModelV1nores": {"class": TransformerModelV1nores, "alias": "TV1nr"},
    "TransformerModelV1noresForceDiag": {
        "class": TransformerModelV1noresForceDiag,
        "alias": "TV1nrFD",
    },
    "TransformerModelV1noresOmitLast": {
        "class": TransformerModelV1noresOmitLast,
        "alias": "TV1nrOL",
    },
    "TransformerModelV1noresForceDiagAndOmitLast": {
        "class": TransformerModelV1noresForceDiagAndOmitLast,
        "alias": "TV1nrFDOL",
    },
    "TransformerModelV2": {"class": TransformerModelV2, "alias": "TV2"},
    "TransformerModelV2nores": {"class": TransformerModelV2nores, "alias": "TV2nr"},
    "TransformerModelV2noresOmitLast": {
        "class": TransformerModelV2noresOmitLast,
        "alias": "TV2nrOL",
    },
    "TransformerModelQKVnores": {"class": TransformerModelQKVnores, "alias": "TQKVnr"},
}
# define companion dict mapping alias to class string
MODEL_CLASS_ALIAS_TO_STR = {v["alias"]: k for k, v in MODEL_CLASS_FROM_STR.items()}


if __name__ == "__main__":
    # sequence: three vectors from R^2
    sample_sequence = np.array([[[1, 1, 1], [2, 3, 4]]]).astype("float32")

    print("Prep sample input for each model:")
    print("=" * 20)
    sample_input = torch.tensor(sample_sequence).to(
        device
    )  # add a batch dimension to front

    print("sample_input.size()", sample_input.size())
    batchsz, n_dim, n_tokens = sample_input.size()

    # Demonstrate parameter count table and sample outputs
    print("\nModel (sequence): ReturnLastToken()")
    model = ReturnLastToken()
    model.to(device)
    count_parameters(model)
    print("sample_output:", model(sample_input))

    print(sample_input)
    print(torch.transpose(sample_input, 1, 2))
    print("the device of sample input on line 1274", sample_input.device)

    print("\nModel (sequence): TransformerModelV1()")
    model = TransformerModelV1(n_tokens, n_dim)
    model.to(device)
    count_parameters(model)

    print("sample_output:", model(sample_input))

    # print('\nModel (sequence): TransformerModelV2()')
    # model = TransformerModelV2(n_tokens, n_dim)
    # model.to(device)
    # count_parameters(model)
    # print('sample_output:', model(sample_input))

    # # ========= scrap
    # print('\n========= scrap and sampleinput device', sample_input.device)
    # # emulate forwards pass of linear self-attention (Bartlett 2023, Eq. 3)
    # W_PV = torch.randn(2, 2).to(device)
    # W_KQ = torch.randn(2, 2).to(device)
    # rho = n_tokens

    # print(type(W_PV), W_PV.dtype)
    # print(type(W_KQ), W_KQ.dtype)
    # print(type(sample_input), sample_input.dtype)

    # attn_arg = torch.transpose(sample_input, 1,2).to(device) @ W_KQ @ sample_input / rho
    # f_attn = sample_input + W_PV @ sample_input @ attn_arg

    # print(f_attn)
    # out = f_attn[:, :, -1]  # take dim_n output result at last token, for all batches
    # print(out, out.size())
    # print(out.flatten(), out.flatten().size())

    # print('\nSynthesize train/test data for sequence models')
    # print('='*20)
    # x_train, y_train, x_test, y_test, _, _ = data_train_test_split_linear(
    #     context_len=10, # was 100
    #     dim_n=8,
    #     num_W_in_dataset=100, #was 1000
    #     context_examples_per_W=1,
    #     test_ratio=0.2,
    #     verbose=True,
    #     seed=0)
    # print('x_train.shape:', x_train.shape, x_train.device) #by default in this exploration device is CPU

    # print('\nSample input -> output for ReturnLastToken() sequence model')
    # print('='*20)
    # model = ReturnLastToken()

    # single_batch = x_train[0, :, :].unsqueeze(0)  # add back trivial batch dim
    # print('\tsingle_batch tensor.size:', single_batch.shape)
    # out_from_single_batch = model(single_batch)
    # print('\tout_from_single_batch tensor.size:', out_from_single_batch.shape)

    # full_batch = x_train[:, :, :]
    # print('\n\tfull_batch tensor.size:', full_batch.shape)
    # out_from_full_batch = model(full_batch)
    # print('\tout_from_full_batch tensor.size:', out_from_full_batch.shape)

    # print('\nSample input -> output for TransformerModelV1() sequence model')
    # print('='*20)
    # model = TransformerModelV1(100, 8) #8 was 128

    # print("last model to device which is", device) #issue put model on device when device is mps
    # model.to(device)

    # single_batch = x_train[0, :, :].unsqueeze(0)
    # print('\tsingle_batch tensor.size:', single_batch.shape)
    # out_from_single_batch = model(single_batch)
    # print('\tout_from_single_batch tensor.size:', out_from_single_batch.shape)

    # full_batch = x_train[:, :, :]
    # print('\n\tfull_batch tensor.size:', full_batch.shape)
    # out_from_full_batch = model(full_batch)

    # print('\tout_from_full_batch tensor.size:', out_from_full_batch.shape)

    # print('\nSample input -> output for TransformerModelV2() sequence model')
    # print('='*20)
    # model = TransformerModelV2(100, 8) #was 128

    # single_batch = x_train[0, :, :].unsqueeze(0)
    # print('\tsingle_batch tensor.size:', single_batch.shape)
    # out_from_single_batch = model(single_batch)
    # print('\tout_from_single_batch tensor.size:', out_from_single_batch.shape)

    # full_batch = x_train[:, :, :]
    # print('\n\tfull_batch tensor.size:', full_batch.shape)
    # out_from_full_batch = model(full_batch)

    # print("full_batch", full_batch) #values change with each run

    # print('\tout_from_full_batch tensor.size and device:', out_from_full_batch.shape, out_from_full_batch.device)

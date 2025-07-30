import os
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


from data_gen_denoise import (
    gen_gaussian_data,
    gen_uniform_data,
    sample_from_subspace,
    proj_affine_subspace_estimator,
)


class DatasetWrapper(Dataset):
    """
    (relic): currently, there is a "remainder" batch at the end, with size smaller than batch_size -- could discard it
    """

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

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


def report_dataset_loss(
    net, criterion, dataloader, data_label, device, print_val=False, plot_some=False
):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data
            outputs = net(samples.to(device))

            if plot_some:
                # visualization and debugging
                if count < 5:
                    plt.plot(outputs, label="pred")
                    plt.plot(targets, label="true")
                    plt.legend()
                    plt.show()

            mse_loss_total += criterion(
                outputs.to(device), targets.to(device)
            ).item()  # add the *mean* MSE for the batch to mse_loss_total

            # Note: we could do count += bsz, but we assume batch-mean inside criterion call
            # e.g. count += samples.size(0)
            #  - this would resolve edge issue when dataset (e.g. test set) not divisible by bsz
            count += 1  # this +1 and divide by count assumes batch-mean inside criterion call, as is done above

    mse_loss = mse_loss_total / count
    if print_val:
        print("\t%s data loss: %.3e (batches=%d)" % (data_label, mse_loss, count))
    return mse_loss


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
    style_subspace_dimensions="random",  # DATAGEN_GLOBALS[0]['style_subspace_dimensions'],
    style_origin_subspace=True,  # DATAGEN_GLOBALS[0]['style_origin_subspace'],      # TODO convert to float  0 < m < 1 magnitude
    style_corruption_orthog=False,  # DATAGEN_GLOBALS[0]['style_corruption_orthog'],  # TODO convert to float  0 < a < 1
    sigma2_corruption=0.5,  # DATAGEN_GLOBALS[0]['sigma2_corruption'],
    sigma2_pure_context=2.0,  # DATAGEN_GLOBALS[0]['sigma2_pure_context'],
    corr_scaling_matrix=None,  # DATAGEN_GLOBALS[0]['corr_scaling_matrix'],
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

    (
        x_train,
        y_train,
        x_test,
        y_test,
        train_data_subspaces,
        test_data_subspaces,
    ) = data_train_test_split_util(
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

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


def data_train_test_split_linear_noise_drift(
    context_len=100,
    dim_n=128,
    num_W_in_dataset=100,
    context_examples_per_W=1,
    samples_per_context_example=1,
    test_ratio=0.2,
    verbose=True,
    seed=None,
    style_subspace_dimensions=DATAGEN_GLOBALS[4]["style_subspace_dimensions"],
    style_origin_subspace=DATAGEN_GLOBALS[4][
        "style_origin_subspace"
    ],  # TODO convert to float  0 < m < 1 magnitude
    style_corruption_orthog=DATAGEN_GLOBALS[4][
        "style_corruption_orthog"
    ],  # TODO convert to float  0 < a < 1
    sigma2_corruption_train=DATAGEN_GLOBALS[4]["sigma2_corruption_train"],
    sigma2_corruption_test=DATAGEN_GLOBALS[4]["sigma2_corruption_test"],
    sigma2_pure_context=DATAGEN_GLOBALS[4]["sigma2_pure_context"],
    corr_scaling_matrix=DATAGEN_GLOBALS[4]["corr_scaling_matrix"],
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
    print("\tsigma_corruption train:", sigma2_corruption_train)
    print("\tsigma_corruption test:", sigma2_corruption_test)
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
    corruption_kicks_train_list = []
    corruption_kicks_test_list = []

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
            corruption_cov = sigma2_corruption_train * np.eye(
                dim_W_perp
            )  # TODO replace by A @ A.T, given full rank A?
            c = gen_gaussian_data(
                np.zeros(dim_W_perp), corruption_cov, (context_examples_per_W), rng=rng
            )  # random coefficients for the basis {v_i}; shape samples x dim_d
            corruption_kicks = W_random_orthog_direction @ c.T
        else:
            corruption_mean = np.zeros(dim_n)
            corruption_cov_train = sigma2_corruption_train * np.eye(dim_n)

            print("sigma2_corruption_train", sigma2_corruption_train)
            print("sigma2_corruption_test", sigma2_corruption_test)

            corruption_cov_test = sigma2_corruption_test * np.eye(dim_n)

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
            corruption_kicks_train = gen_gaussian_data(
                corruption_mean, corruption_cov_train, context_examples_per_W, rng=rng
            )  # shape samples x dim_n
            corruption_kicks_train = (
                corruption_kicks_train.T
            )  # shape dim_n x context_examples_per_W

            corruption_kicks_test = gen_gaussian_data(
                corruption_mean, corruption_cov_test, context_examples_per_W, rng=rng
            )  # shape samples x dim_n
            corruption_kicks_test = (
                corruption_kicks_test.T
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
            corruption_kicks_for_sample_train = corruption_kicks_train[
                :, sample_idx
            ]  # gen both kinds for all samples
            corruption_kicks_for_sample_test = corruption_kicks_test[:, sample_idx]

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
            # X_sequence[:, -1] = y_target + corruption_kicks_for_sample

            x_total.append(X_sequence)  # without the noise here
            y_total.append(y_target)

            corruption_kicks_train_list.append(
                corruption_kicks_for_sample_train
            )  # add later in train test split
            # print("what is the shape of corruption_kicks_for_sample_train ", corruption_kicks_for_sample_train.shape)

            corruption_kicks_test_list.append(corruption_kicks_for_sample_test)

            data_subspace_dict[j]["W_m"] = rand_m
            data_subspace_dict[j]["W_V"] = rand_V
            data_subspace_dict[j]["W_dim"] = dim_d
            j += 1

    (
        x_train,
        y_train,
        x_test,
        y_test,
        train_data_subspaces,
        test_data_subspaces,
    ) = data_train_test_split_util_noise_drift(
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
        corruption_kicks_for_sample_test=corruption_kicks_test_list,
        corruption_kicks_for_sample_train=corruption_kicks_train_list,
    )

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


def reload_lossbaseline_dict(dir_replot):
    with open(dir_replot + os.sep + "loss_baselines.txt") as f:
        loss_baselines_dict = {}
        lines = [a.strip() for a in f.readlines()]
        for lstr in lines:
            a, b_str = lstr.split(",")
            loss_baselines_dict[a] = float(b_str)
        return loss_baselines_dict

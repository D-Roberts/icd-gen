import datetime
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle  # must we? TODO: get rid of the pickle later
import torch


# function to project onto d-dim affine subspace W(m, V) of R^n
def proj_affine_subspace(W_m, W_V, x):
    """
    Projects a vector x in R^n onto the d-dim affine subspace W specified by W_m (n x 1) and W_V (n x d)
    """
    assert x.shape == W_m.shape
    projector = W_V @ np.linalg.inv(W_V.T @ W_V) @ W_V.T
    return W_m + projector @ (x - W_m)


def subspace_offset_and_basis_estimator(X_seq, eval_cutoff=1e-4, verbose_vis=False):
    """
    See proj_affine_subspace_estimator
        - functionalized the first part of that original function
    """
    ndim, ncontext = X_seq.shape
    n_samples = ncontext  # alias

    mu = np.expand_dims(np.mean(X_seq, axis=1), axis=1)

    # Manual PCA
    A = (X_seq - mu) @ (X_seq - mu).T / n_samples

    eig_D, eig_V = np.linalg.eig(A)
    sorted_indexes = np.argsort(eig_D)
    eig_D = eig_D[sorted_indexes]
    eig_V = eig_V[:, sorted_indexes]

    indices_nonzero_evals = np.where(eig_D > eval_cutoff)[0]
    nonzero_evecs = eig_V[:, indices_nonzero_evals]
    nonzero_evals = eig_D[indices_nonzero_evals]

    # Showcase the de-corruption (by estimating offset and basis from the sequence)
    estimate_offset = mu
    estimate_basis = np.real(nonzero_evecs)
    estimate_evals = np.real(nonzero_evals)

    if verbose_vis:
        print(estimate_offset.shape, estimate_basis.shape)
        print(len(indices_nonzero_evals), "vs", eig_D.shape)
        print("...")
        plt.figure()
        plt.plot(eig_D, "-ok", markersize=3)
        plt.plot(
            indices_nonzero_evals, eig_D[indices_nonzero_evals], "-or", markersize=3
        )
        plt.title(
            "eigenvalues from PCA on X_seq (num above = %d)"
            % len(indices_nonzero_evals)
        )
        plt.xlabel(r"rank $i$")
        plt.ylabel(r"$\lambda_i$")
        plt.axhline(eval_cutoff, color="k", linestyle="--")
        plt.show()

    return estimate_offset, estimate_basis, estimate_evals


def proj_affine_subspace_estimator(
    X_seq, x_corrupt, eval_cutoff=1e-4, verbose_vis=False, return_basis=False
):
    """
    See transformer_A.ipynb

    Args:
        X_seq     - 2-d sequence X_seq of shape ndim, ncontext - 1
        x_corrupt - 1-d arr of shape ndim

    Note:
        together X_seq and x_corrupt compose the input sequence to the NN (x_corrupt is suffix, goal is to decorrupt)
    """
    # step 1
    est_offset, est_basis, _ = subspace_offset_and_basis_estimator(
        X_seq, eval_cutoff=eval_cutoff, verbose_vis=verbose_vis
    )
    # step 2
    projection_b_onto_W = proj_affine_subspace(est_offset, est_basis, x_corrupt)

    if verbose_vis:
        print(est_offset.shape, est_basis.shape, x_corrupt.shape)
        print(projection_b_onto_W.shape)

    if return_basis:
        return projection_b_onto_W, est_basis
    else:
        return projection_b_onto_W


def modified_bessel_firstkind_scipy(n, z):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv
    n is the order; must be integer
    z is the argument; must be real

    In case of overflow (large z), try https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html#scipy.special.ive
    "exponentially scaled modified Bessel function of the first kind"
    """
    return iv(n, z)


def modified_bessel_firstkind_scipy_expscale(n, z):
    """
    In case of overflow (large z), try https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html#scipy.special.ive
    "exponentially scaled modified Bessel function of the first kind"

    ive(n, z) = iv(n, z) * exp(-abs(z.real))
    """
    return ive(n, z)


def bessel_ratio_np1_over_n(n, z):
    """
    use scipy ive for numeric stability; note the exp scalings cancel out
    """
    return modified_bessel_firstkind_scipy_expscale(
        n + 1, z
    ) / modified_bessel_firstkind_scipy_expscale(n, z)


def bessel_ratio_subhalf_sub3half(n, z):
    """
    use scipy ive for numeric stability; note the exp scalings cancel out
    - n = 2 case corresponds to sphere in R3, so I_{3/2} / I_{ 1/2}
    - n = 1 case corresponds to circle in R2, so I_{1/2} / I_{-1/2}
    """
    return modified_bessel_firstkind_scipy_expscale(
        n - 0.5, z
    ) / modified_bessel_firstkind_scipy_expscale(n - 1.5, z)


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

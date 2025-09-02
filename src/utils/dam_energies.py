import sys
import os
import numpy as np
import torch


# these are from the Incontext Denoise Code.
# TODO@DR: torchify and debug. Use efficient torch implementations where available.

"""
Torch implementations for energy.
"""

# def logsumexp(z):
#     """z is a 1D numpy array
#     Computes the log of the sum of exponentials of input elements.
#     Also for stabiltiy, subtracts the maximum value from z before exponentiation.
#     """
#     z_max = np.max(z)
#     return z_max + np.log(np.sum(np.exp(z - z_max)))

# just use torch.logsumexp


# def softmax(z):
#     """z is a 1D numpy array
#     Computes the softmax of the input elements.
#     """
#     e_z = np.exp(z - np.max(z))  # for numerical stability
#     return e_z / np.sum(e_z)

# just use torch.softmax

# plan to use the net prediction on the fly


# TODO@DR: calculate in batch
def energy(s, X, c_lambda, beta, c_k):
    """from m.smart paper not the hopfield"""
    term1 = 0.5 * c_lambda * torch.norm(s, p=2, dim=-1) ** 2  # norm squared in energy
    scores = beta * c_k * (X.T @ s)
    # print(f"scores shape {scores.shape}") #4, 5, 4

    lse = torch.logsumexp(scores, dim=0)
    term2 = (1 / (beta * c_k)) * lse
    # torch has logsumexp
    # print(f"logsumexp(scores) {lse}")
    return term1 - term2


def grad_energy(s, X, c_lambda, beta, c_k):
    """s is the query; the noised last token vector"""
    scores = beta * c_k * (X.T @ s)
    a = torch.softmax(scores, dim=1)  # so the variant with the softmax
    # print("debug a shape in grad_energy*******:", a.shape)
    return c_lambda * s - X @ a


# adhoc test
L = 5  # context
sig2_noise = 1  # check for 0.1 and 2 which I have in the drift /shift - showing a lack
# of generalization when test noise diff from train noise to see what the energy
# looks like

beta = 1.0
ck = 1.0 / sig2_noise
cv = 1.0
c_lambda = 1 / cv  # lambda is 1/alpha

gamma_step = 1.0 * cv

X1L = torch.randn(4, 5)  # dim, context; no batch here
s = torch.randn(4, 1)
print(f"shape of s {s.shape}")

en = energy(s, X1L, c_lambda, beta, ck)
print(f" en and its shape {en} and {en.shape}")

grad = grad_energy(s, X1L, c_lambda, beta, ck)
print(f"grad {grad} and its shape {grad.shape}")
assert grad.shape == s.shape  # yes ok

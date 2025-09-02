import sys
import os
import numpy as np


# these are from the Incontext Denoise Code.
# TODO@DR: torchify and debug. Use efficient torch implementations where available.


def logsumexp(z):
    """z is a 1D numpy array
    Computes the log of the sum of exponentials of input elements.
    Also for stabiltiy, subtracts the maximum value from z before exponentiation.
    """
    z_max = np.max(z)
    return z_max + np.log(np.sum(np.exp(z - z_max)))


def softmax(z):
    """z is a 1D numpy array
    Computes the softmax of the input elements.
    """
    e_z = np.exp(z - np.max(z))  # for numerical stability
    return e_z / np.sum(e_z)


# plan to use the net prediction on the fly


def energy(q, X, c_lambda, beta, c_k):
    """from m.smart paper not the hopfield"""
    term1 = 0.5 * c_lambda * np.sum(q**2)
    scores = beta * c_k * (X.T @ q)
    term2 = (1 / (beta * c_k)) * logsumexp(scores)
    return term1 - term2


def grad_energy(q, X, c_lambda, beta, c_k):
    """q is the query; the noised last token vector"""
    scores = beta * c_k * (X.T @ q)
    a = softmax(scores)  # so the variant with the softmax
    # print("debug a shape in grad_energy*******:", a.shape)
    return c_lambda * q - X @ a

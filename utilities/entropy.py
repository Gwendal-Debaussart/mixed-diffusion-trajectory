import numpy as np


def singular_entropy(P):
    """
    Computes the singular entropy of a matrix.
    """
    singular_vals = np.linalg.svd(P, compute_uv=False)
    singular_vals = np.abs(singular_vals)
    singular_vals = singular_vals[singular_vals != 0]
    singular_vals /= np.linalg.norm(singular_vals)
    return -np.sum(singular_vals * np.log(singular_vals))


def spectral_entropy(P):
    """
    Computes the spectral entropy of a matrix.
    """
    eigvals = np.linalg.eigvals(P)
    eigvals = np.abs(eigvals)
    eigvals = eigvals[eigvals != 0]
    eigvals = eigvals / np.linalg.norm(eigvals)
    return -np.sum(eigvals * np.log(eigvals))

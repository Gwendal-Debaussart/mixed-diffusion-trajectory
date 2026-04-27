import numpy as np


def entropy_from_values(values: np.ndarray) -> float:
    """
    Computes Shannon entropy from a spectrum-like vector.

    Values are converted to non-negative weights, optionally raised to a power,
    normalized by their sum, and evaluated as -sum(p log p).
    """
    values = np.abs(values[values != 0])
    values /= np.linalg.norm(values)
    return -np.sum(values * np.log(values))


def singular_entropy(P):
    """
    Computes the singular entropy of a matrix.
    """
    singular_vals = np.linalg.svd(P, compute_uv=False)
    return entropy_from_values(singular_vals)


def spectral_entropy(P):
    """
    Computes the spectral entropy of a matrix.
    """
    eigvals = np.linalg.eigvals(P)
    return entropy_from_values(eigvals)

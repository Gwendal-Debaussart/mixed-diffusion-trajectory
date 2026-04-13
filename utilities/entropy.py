import numpy as np


def entropy_from_values(values: np.ndarray) -> float:
    """
    Computes Shannon entropy from a spectrum-like vector.

    Values are converted to non-negative weights, optionally raised to a power,
    normalized by their sum, and evaluated as -sum(p log p).
    """
    weights = np.abs(np.asarray(values, dtype=float))
    weights = weights[weights > 0]
    if weights.size == 0:
        return 0.0

    total = np.sum(weights)
    if total == 0:
        return 0.0

    probs = weights / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


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

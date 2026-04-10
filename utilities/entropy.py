import numpy as np


def entropy_from_values(values: np.ndarray, power: int = 1) -> float:
    """
    Computes entropy from a spectrum-like vector, optionally for powered values.
    """
    v = np.abs(values)
    v = v[v > 0]
    if v.size == 0:
        return 0.0

    # Use log-domain scaling for numerical stability when power is large.
    logs = power * np.log(v)
    max_log = np.max(logs)
    scaled = np.exp(logs - max_log)
    norm = np.linalg.norm(scaled)
    if norm == 0:
        return 0.0

    probs = scaled / norm
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def singular_entropy(P):
    """
    Computes the singular entropy of a matrix.
    """
    singular_vals = np.linalg.svd(P, compute_uv=False)
    return entropy_from_values(singular_vals)


def powered_singular_entropy(singular_vals: np.ndarray, t: int) -> float:
    """
    Computes the singular entropy of P^t from singular values of P.
    """
    return entropy_from_values(singular_vals, power=t)


def powered_spectral_entropy(eigvals: np.ndarray, t: int) -> float:
    """
    Computes the spectral entropy of P^t from eigenvalues of P.
    """
    return entropy_from_values(eigvals, power=t)


def spectral_entropy(P):
    """
    Computes the spectral entropy of a matrix.
    """
    eigvals = np.linalg.eigvals(P)
    return entropy_from_values(eigvals)

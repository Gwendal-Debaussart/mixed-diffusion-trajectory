from functools import reduce
import numpy as np
from kneed import KneeLocator
from utilities import singular_entropy

def alternating_diffusion(X):
    """
    Implementation of the Alternating Diffusion algorithm.

    Parameters:
    ---------
    X: list of np.ndarray
        List of kernel matrices representing different views of the data.
    Returns:
    -------
    K_ad: np.ndarray
        The combined kernel matrix after applying Alternating Diffusion.
    """
    return reduce(lambda a, b: a @ b, X)


def powered_alternating_diffusion(X, power=None):
    """
    Implementation of the Powered Alternating Diffusion algorithm.

    Parameters:
    ---------
    X: list of np.ndarray
        List of kernel matrices representing different views of the data.
    power: int
        The exponent to which the combined kernel matrix is raised.
        if power = 1, it corresponds to the standard Alternating Diffusion.
        if power = None, default is set using elbow of singular values of the alternating diffusion matrix.
    Returns:
    -------
    K_pad: np.ndarray
        The combined kernel matrix after applying Powered Alternating Diffusion.
    """
    P_ad = alternating_diffusion(X)
    if power is None:
        singular_entropies = [
            singular_entropy(np.linalg.matrix_power(P_ad, t)) for t in range(1, 25)
        ]
        kn = KneeLocator(
            range(1, 25), singular_entropies, curve="convex", direction="decreasing"
        )
        if kn.elbow is None:
            print("No elbow found, setting power to 1")
            power = 1
        else:
            power = kn.elbow
    P_pad = np.linalg.matrix_power(P_ad, power)
    return P_pad

from kneed import KneeLocator
import numpy as np
from functools import reduce
from utilities import spectral_entropy

def integrated_diffusion(X):
    """
    Implementation of the Integrated Diffusion algorithm. (Kuchroo et al., 2022)

    Parameters:
    ---------
    X: list of np.ndarray
        List of kernel matrices representing different views of the data.
    Returns:
    -------
    K_id: np.ndarray
        The combined kernel matrix after applying Integrated Diffusion.
    """
    time_range = range(1, 25)
    spectral_entropies = {}
    elbow = {}
    for i in range(len(X)):
        spectral_entropies[i] = [spectral_entropy(np.linalg.matrix_power(X[i], t)) for t in time_range]
        knee_locator = KneeLocator(time_range, spectral_entropies[i], curve="convex", direction="decreasing")
        optimal_time = knee_locator.elbow
        if optimal_time is None:
            print(f"No elbow found for view {i}, setting time to 1")
            optimal_time = 1
        elbow[i] = optimal_time
    X_powered = [np.linalg.matrix_power(X[i], elbow[i]) for i in range(len(X))]
    return reduce(lambda a, b: a @ b, X_powered)
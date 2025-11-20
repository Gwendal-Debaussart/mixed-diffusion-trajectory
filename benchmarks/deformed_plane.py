import numpy as np


def deformed_plane(n_samples=3000, random_state=42):
    """
    Generate the Deformed plane dataset.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
    random_state : int
        Random seed for reproducibility. Default is 42.
    Returns:
    --------
    x : list of np.ndarray
        List containing two arrays of shape (n_samples, 3), each representing a deformed plane.
    Z : np.ndarray
        Array of shape (n_samples, 1) representing the parameterization of the deformed plane. (Useful only for visualization as 1D colorbar)
    """
    np.random.seed(random_state)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    h = 21 * np.random.rand(n_samples)
    Z = np.vstack([t, h]).T

    X1 = np.zeros((n_samples, 3))
    X1[:, 0] = t * np.cos(0.65 * t)
    X1[:, 1] = 0.2 * h + 0.3 * np.sin(t)
    X1[:, 2] = t * np.sin(t)

    X2 = np.zeros((n_samples, 3))
    X2[:, 0] = h + np.sin(2 * t) + 0.3 * np.cos(h)
    X2[:, 1] = t + np.sin(h) + 0.3 * np.cos(2 * t)
    X2[:, 2] = np.sin(t) + np.cos(h)

    return [X1, X2], Z[:, 0]

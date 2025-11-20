import numpy as np


def helix_b(num_samples=1500):
    """
    Generate the Helix B dataset (Lindenbaum et al., 2020).

    Parameters:
    -----------
    num_samples : int
        Number of samples to generate.
    num_classes : int, optional
        Number of classes (default: 2).
    sigma : float, optional
        Noise parameter (unused, default: 0.5).
    random_state : int, optional
        Random seed (default: 333).

    Returns:
    --------
    x : list of np.ndarray
        List containing two arrays of shape (num_samples, 3), each representing a helix.
    Y : np.ndarray
        Array of shape (num_samples,) representing the parameterization of the helix.
    """
    x = []
    a = np.linspace(0, 2 * np.pi, num_samples)
    b = (a + 0.5 * np.pi) % (2 * np.pi)
    Y = a
    for i in [a, b]:
        X_temp = np.zeros((num_samples, 3))
        X_temp[:, 0] = np.cos(5 * i)
        X_temp[:, 1] = np.sin(5 * i)
        X_temp[:, 2] = i
        x.append(4 * X_temp)

    return x, Y

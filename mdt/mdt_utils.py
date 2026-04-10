import numpy as np

def mdt_operator(trajectory, X):
    """
    Constructs the mixed-view diffusion trajectory operator from a given trajectory.

    Parameters:
    -----------
    trajectory: np.ndarray of shape (T, k)
        The trajectory of weights for each view at each time step.
    X: list of np.ndarray of shape (n, n)
        List of kernel matrices for each view.

    Returns:
    --------
    W: np.ndarray of shape (n, n)
        The resulting operator after applying the mixed diffusion process.
    """
    # Pre-stack X into a 3D array: shape (num_matrices, n, n)
    X_stack = np.stack(X, axis=0)  # (k, n, n)

    # trajectory: (t, k) — weighted sum at each step: (k,) · (k, n, n) -> (n, n)
    # Compute all weighted sums at once: shape (t, n, n)
    weighted = np.einsum('tk,knm->tnm', trajectory, X_stack)

    W = weighted[0]
    for i in range(1, len(weighted)):
        W = weighted[i] @ W
    return W

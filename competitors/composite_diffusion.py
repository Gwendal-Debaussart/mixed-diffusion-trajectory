import numpy as np

def composite_diffusion_operator(X):
    """
    Constructs the composite diffusion operator for two views by summing the cross-diffusion terms.

    Parameters:
    ----------
    X : list of np.ndarray
        List containing two transition matrices representing different views of the data.

    Returns:
    -------
    W : np.ndarray
        Composite diffusion operator.
    """
    if len(X) == 2:
        return X[0] @ X[1].T + X[1] @ X[0].T
    else:
        raise ValueError("Composite diffusion operator is only defined for two views.")
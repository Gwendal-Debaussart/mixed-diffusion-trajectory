import numpy as np

def multiview_diffusion(X):
    """
    Implementation of the Multi-View Diffusion algorithm. (Lindenbaum et al., 2020)

    Parameters:
    ---------
    X: list of np.ndarray
        List of *unnormalized* kernel matrices representing different views of the data.
    Returns:
    -------
    K_mv: np.ndarray
        The combined kernel matrix after applying Multi-View Diffusion.
    """
    n = len(X)
    m = X[0].shape[0]
    multiview_operator = np.zeros((n*m, n*m))

    for i in range(n):
        for j in range(i + 1, n):
            block = X[i] @ X[j]
            multiview_operator[i*m:(i+1)*m, j*m:(j+1)*m] = block
            multiview_operator[j*m:(j+1)*m, i*m:(i+1)*m] = block.T

    row_sums = multiview_operator.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    multiview_operator /= row_sums
    return multiview_operator

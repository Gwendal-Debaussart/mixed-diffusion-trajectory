import numpy as np


def cross_diffusion_operator(X, t=25):
    """
    Computes the Cross Diffusion operator (Wang et al., 2012).

    Parameters:
    ---------
    X: np.ndarray
        list of transition kernel matrix representing the data.
    t: int
        Number of iterations for the cross diffusion process. Default is 25.
    Returns:
    -------
    K_cd: np.ndarray
        The combined kernel matrix after applying Cross Diffusion.

    Notes:
    -----
    Method based on "Unsupervised metric fusion by cross diffusion" by Wang et al., 2012.
    Base iteration time is set to 25, the original paper didn't state any parameter tuning for it, but states 'large number of iteration are favored'.
    """
    n_views = len(X)
    X_temp = X.copy()
    for _ in range(t):
        total_sum = np.sum(X_temp, axis=0)
        others_avg = (total_sum[None, :, :] - X_temp) / (n_views - 1)
        # Batch matmul: X_i @ others_avg_i @ X_i.T
        X_temp = np.matmul(
            np.matmul(X, others_avg), np.transpose(X, (0, 2, 1))
        )
        # Clipping to avoid numerical issues (not included in original paper)
        X_temp = np.clip(X_temp, -1e6, 1e6)

    return np.mean(X_temp, axis=0)

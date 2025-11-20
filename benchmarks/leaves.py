from .utilities import load_mat_file


def leaves():
    """
    Load the 101 Leaves dataset.

    Returns:
    --------
    list_views : list of np.array
        List containing the kernel matrices for each view.
    true_labels : np.array
        Array of true labels for the data points.

    References:
    -----------
    .mat file obtained from
    https://github.com/ChuanbinZhang/Multi-view-datasets/
    """
    X_views, y = load_mat_file("100leaves")
    return X_views, y
from .utilities import load_mat_file


def caltech101_7():
    """
    Load the Caltech101-7 dataset.

    Returns:
    --------
    list_views : list of np.array
        List containing the kernel matrices for each view.
    true_labels : np.array
        Array of true labels for the data points.
    """
    X_views, y = load_mat_file("caltech101-7")
    return X_views, y
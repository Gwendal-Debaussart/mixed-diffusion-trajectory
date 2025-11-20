import numpy as np
from .utilities import load_mat_file
from sklearn.decomposition import PCA

def yale():
    """
    Load the Yale Faces dataset.
    do a PCA set:
    Intensity(4096), LBP(3304), Gabor(6750)
    """
    X_views, y = load_mat_file("Yale")
    X_preproc = []
    for x in X_views:
        if x.shape[1] > 100:
            x_pre = PCA(n_components=100).fit_transform(x)
            x_pre /= np.linalg.norm(x_pre, axis=1, keepdims=True) + 1e-9
        else:
            x_pre = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        X_preproc.append(x_pre)
    return X_preproc, y

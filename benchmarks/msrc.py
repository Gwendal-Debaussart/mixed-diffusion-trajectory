from .utilities import load_mat_file
from sklearn.decomposition import PCA
import numpy as np


def msrc():
    """
    Load the MSRC dataset.
    CM(24), HOG(576), GIST(512), LBP(256), CENT(254)

    PCA on HOG GIST LBP CENT to 100 dims each
    5 views total
    7 classes
    240 samples
    10 samples per class per view
    2 views are image features, 3 views are text features
    """
    X_views, y = load_mat_file("MSRC-v5")
    X_preproc = []
    for x in X_views:
        if x.shape[1] > 100:
            x_pre = PCA(n_components=100).fit_transform(x)
            x_pre /= np.linalg.norm(x_pre, axis=1, keepdims=True) + 1e-9
        else:
            x_pre = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        X_preproc.append(x_pre)

    return X_preproc, y

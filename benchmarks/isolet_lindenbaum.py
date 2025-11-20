import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors


def isolet_lindenbaum():
    """
    Load the ISOLET dataset and compute three different kernel matrices as in
    Lindenbaum et al., 2019.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "source/isolet")
    X_path = os.path.join(base_dir, "isolet_features.csv")
    y_path = os.path.join(base_dir, "isolet_targets.csv")
    X = pd.read_csv(X_path, index_col="id")
    true_labels = pd.read_csv(y_path)["class"]
    X = np.array(X)[:1600]
    true_labels = np.array(true_labels, dtype=int)[:1600].flatten()

    list_views = []
    n_neighbor = int(np.floor(np.log(len(X))))
    d = distance.cdist(X, X)
    bandwidth = np.max([i[i > 0].min() for i in d])
    distance_matrix = squareform(pdist(X, "euclidean"))
    nbrs = NearestNeighbors(n_neighbors=n_neighbor, metric="precomputed").fit(
        distance_matrix
    )
    k_nn = nbrs.kneighbors_graph(distance_matrix, mode="distance").toarray()
    k_temp = np.copy(k_nn)
    # K_1 and K_2
    k_temp[np.nonzero(k_nn)] = np.exp(
        -k_nn[np.nonzero(k_nn)] ** 2 / (2 * (bandwidth**2))
    )
    list_views.append(k_temp)
    k_temp[np.nonzero(k_nn)] = np.exp(-k_nn[np.nonzero(k_nn)] / bandwidth)
    list_views.append(k_temp)
    # K_3
    X_tilde = X - X.mean(axis=1, keepdims=True)
    X_normed = X_tilde / np.linalg.norm(X_tilde, axis=1, keepdims=True)
    T = X_normed @ X_normed.T
    K3 = np.exp((T - 1) / (2 * bandwidth**2))
    k_nn = np.zeros_like(K3)
    for i in range(K3.shape[0]):
        idx = np.argsort(K3[i])[::-1][1 : n_neighbor + 1]
        k_nn[i, idx] = K3[i, idx]
    list_views.append(k_nn)

    list_views = [(k + k.T) / 2 for k in list_views]
    return list_views, true_labels

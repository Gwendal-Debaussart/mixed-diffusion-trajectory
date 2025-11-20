import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import os
import scipy.io


def get_kernel_matrix(
    x: np.array,
    n_neighbor: int = None,
    bandwidth: float = None,
    metric: str = "euclidean",
    normalize: bool = True,
):
    """
    Parameters
    ----------
    x: np.array
      matrix
    n_neighbor: int
      number used in the k-n graph part
    bandwidth : real
      the bandwidth used for the exponential kernel
    metric: str
      the metric used ("minkowski","cosine", or custom)
    normalize: bool
      whether to normalize the kernel matrix to a row-stochastic matrix

    Returns
    -------
    np.array
      the kernel matrix

    Description
    --------
    The kernel matrix is computed as follows:
    1. Compute the pairwise distance matrix using the specified metric.
    2. If bandwidth is not provided, set it to the minimum non-zero distance.
    3. If n_neighbor is not provided, set it to log(number of samples).
    4. Construct a k-nearest neighbors graph based on the distance matrix.
    5. Apply the exponential kernel to the distances in the k-NN graph.
    6. Symmetrize the kernel matrix.
    7. If normalize is True, convert the kernel matrix to a row-stochastic matrix
    """
    distance_matrix = squareform(pdist(x, metric))
    if bandwidth == None:
        bandwidth = np.max([i[i > 0].min() for i in distance_matrix])
    if n_neighbor == None:
        n_neighbor = int(np.floor(np.log(len(x))))

    nbrs = NearestNeighbors(n_neighbors=n_neighbor, metric="precomputed").fit(
        distance_matrix
    )
    k_nn = nbrs.kneighbors_graph(distance_matrix, mode="distance")
    k_nn.data = np.exp(-k_nn.data**2 / bandwidth)
    k_nn = (k_nn + k_nn.T) / 2
    # the order of the .toarray matters
    if normalize:
        row_sums = k_nn.sum(axis=1)
        row_sums[row_sums == 0] = 1
        k_nn = k_nn / row_sums
    return k_nn.toarray()


def load_mat_file(name: str):
    """
    Load a .mat file and return its contents.
    """
    file_path = os.path.join(
        os.path.dirname(__file__), "source/mat_files/", f"{name}.mat"
    )
    mat = scipy.io.loadmat(file_path)

    X = mat["X"]
    y = mat["y"].ravel()
    if isinstance(X, np.ndarray) and X.dtype == object:
        X_views = [np.array(X[i, 0]) for i in range(X.shape[0])]
    else:
        # Sometimes X itself is already a list/array of 2D matrices
        X_views = [np.array(v) for v in X]
    return X_views, y

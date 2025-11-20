import numpy as np
import sklearn.datasets as skds
import sklearn.preprocessing as pre
from sklearn.decomposition import PCA

def mnist_kuchroo(noise_factor=0.5, random_state=3333):
    """
    Import the MVMnist considered in Kuchroo et al. (2022).

    Parameters:
    -----------
    noise_factor : float
        Standard deviation of the Gaussian noise added to the second view. Default is 0.5
    random_state : int
        Random seed for reproducibility. Default is 3333.

    Returns:
    --------
    x : list of np.ndarray
        List containing two arrays of shape (num_samples, num_features), each representing a view.
    Y : np.ndarray
        Array of shape (num_samples,) representing the digit labels.
    """
    np.random.seed(random_state)
    num = 6000
    ds = skds.fetch_openml("mnist_784")
    x_lat = ds.data[:num]
    x_lat = pre.MinMaxScaler().fit_transform(x_lat)
    Y = np.array(ds.target[:num], dtype=int)
    X1 = x_lat + np.random.normal(0, 0.3, size=x_lat.shape)
    X1 = np.array(np.clip(X1, 0, 1))
    np.random.seed(random_state + 2)
    X2 = x_lat + np.random.normal(0, noise_factor, size=x_lat.shape)
    X2 = np.array(np.clip(X2, 0, 1))

    # Dimensionality reduction and normalization of the views.
    X1 = PCA(n_components=100).fit_transform(X1)
    X2 = PCA(n_components=100).fit_transform(X2)
    X1 /= np.linalg.norm(X1, axis=1, keepdims=True) + 1e-9
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True) + 1e-9
    x = [X1, X2]
    return x, Y

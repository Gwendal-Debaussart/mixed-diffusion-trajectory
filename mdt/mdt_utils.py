import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from utilities.internal_criterions import supervised_metric_list, unsupervised_metric_list
from sklearn.cluster import KMeans

def mdt_operator(trajectory, X):
    """
    Constructs the mixed-view diffusion trajectory operator from a given trajectory.
    """
    t, _ = trajectory.shape
    W = np.eye(X[0].shape[0])
    for i in range(t):
        W =  sum(a * P for a, P in zip(trajectory[i], X)) @ W
    return W

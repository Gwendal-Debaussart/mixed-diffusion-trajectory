from gob.optimizers import Direct
from gob import create_bounds
import numpy as np
from .mdt_utils import mdt_operator
from scipy.special import softmax
from utilities.evaluate import *
from utilities.internal_criterions import *


def bayes_param_to_trajectory(params, X):
    """
    Convert Bayesian optimization parameters to a softmax-normalized trajectory.
    """
    p_values = params.reshape(-1, len(X))
    return softmax(p_values, axis=1)

def mdt_operator_from_params(params, X):
    """
    Create an MDT operator from Bayesian optimization parameters.
    """
    trajectory = bayes_param_to_trajectory(params, X)
    return mdt_operator(trajectory, X)

def mdt_direct(X, t, k, dim_embedding= None, metric = "chs"):
    """
    Perform Mixed Diffusion Trajectories (MDT) optimization. Uses the DIRECT algorithm to find the optimal trajectory parameters that maximize the clustering quality as measured by CH.

    Parameters:
    ----------
    X : list of np.ndarray
        List of transition matrices representing different views of the data.
    t : int
        Length of the trajectory.
     k : int
        Number of clusters for clustering evaluation.
    dim_embedding : int
        Dimension of the embedding space.
    metric : str
        Clustering evaluation metric to optimize (e.g., "chs" for Calinski-Harabasz Score).

    Returns:
    -------
    np.ndarray
        The optimized MDT operator of shape (n_samples, n_samples).
    """
    if dim_embedding is None:
        dim_embedding = k
    bounds = create_bounds(len(X) * t, 0, 20)
    opt = Direct(bounds, n_eval=100)
    f_ = lambda params: evaluate_operator(
        operator=mdt_operator_from_params(params, X),
        X_views=X,
        true_labels=None,
        metric=metric,
        n_clusters=k,
        embedded=False,
        method="svd",
        n_components=dim_embedding,
    )
    a, _ = opt.maximize(f_)
    return mdt_operator_from_params(np.array(a), X)

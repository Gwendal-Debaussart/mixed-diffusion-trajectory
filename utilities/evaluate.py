from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.cluster import KMeans
from .internal_criterions import (
    supervised_metric_list,
    unsupervised_metric_list,
    metric_functions,
)


def get_embedding(P: np.array, dim_embedd: int, method: str = "svd"):
    """
    Obtain the embedding of the given matrix using the specified method.

    Parameters:
    -----------
    P : np.ndarray
        Matrix to embed (e.g., operator or affinity matrix).
    dim_embedd : int
        Dimension of the embedding.
    method : str, optional
        Embedding method to use (default: "svd").
          - "svd": Use Singular Value Decomposition (SVD) to obtain the embedding.
          - "eigen": Use eigenvalue decomposition to obtain the embedding.
          - "truncated_svd": Use Truncated SVD for dimensionality reduction.
          - "precomputed": Return the input matrix P as the embedding without modification. (Used for methods that produce embeddings for consistency in the evaluation pipeline.)

    Returns:
    --------
    np.ndarray
        Embedded representation of the input matrix.
    """
    if method == "svd":
        P_decomp, S, _ = np.linalg.svd(P)
        P_decomp =  P_decomp[:, 1:dim_embedd+1] @ np.diag(S[1:dim_embedd+1])
    elif method == "eigen":
        eigvals, eigvecs = np.linalg.eig(P)
        idx = np.argsort(eigvals)[::-1][1:dim_embedd+1]
        P_decomp = (eigvecs[:, idx] @ np.diag(eigvals[idx])).real
    elif method == "truncated_svd":
        svd = TruncatedSVD(n_components=dim_embedd)
        P_decomp = svd.fit_transform(P)
    elif method == "precomputed":
        P_decomp = P
    else:
        raise ValueError(f"Decomposition method '{method}' not recognized.")

    return P_decomp


def get_clustering(P, num_clusters: int):
    """
    Obtain the clustering of the given matrix using the specified method.

    Parameters:
    -----------
    P : np.ndarray
        Matrix to cluster (e.g., operator or embedding).
    k : int
        Number of clusters.

    Returns:
    --------
    np.ndarray
        Cluster labels for each sample.
    """
    Y = KMeans(n_clusters=num_clusters).fit(P)
    return Y.labels_


def evaluate_operator(
    operator,
    X_views,
    true_labels,
    metric: str,
    n_clusters,
    embedded=False,
    method="svd",
    dim_embedd=10,
):
    """
    Evaluates the given operator on the dataset using the specified metric function.

    Parameters:
    -----------
    operator : np.ndarray
        The operator matrix to evaluate.
    dataset : list of np.ndarray
        The multi-view dataset.
    true_labels : np.ndarray
        The ground truth labels for the data points.
    metric : str
        The metric function to use for evaluation.
    n_clusters : int
        The number of clusters to form.
    embedded : bool, optional
        If True, the operator is already in embedded form. Default is False.
    method : str, optional
        The embedding method to use if not embedded. Default is "svd".
    dim_embedd : int, optional
        The dimension of the embedding if decomposition is needed. Default is 10.

    Returns:
    --------
    float
        The evaluation score from the metric function.

    Note:
    -----
    This function is *not* used in the main benchmark loop. It is provided for easy evaluation of operators during development and debugging.
    """
    if embedded:
        embedding = operator
    else:
        embedding = get_embedding(operator, dim_embedd, method)
    k = n_clusters

    # Ensure embedding has the same number of samples as true_labels and X_views (For MVDM)
    n_samples = len(X_views[0])
    if isinstance(embedding, np.ndarray) and embedding.shape[0] > n_samples:
        embedding = embedding[:n_samples, :]

    y_pred = get_clustering(embedding, k)
    return evaluate_labels(true_labels, X_views, y_pred, metric)


def evaluate_labels(true_labels, X_views, pred_labels, metric):
    """
    Evaluate the given labels using the specified metric.

    Parameters:
    -----------
    true_labels: np.ndarray
        The ground truth labels for the data points.
    X_views: list of np.ndarray
        The multi-view dataset, used for unsupervised metrics.
    pred_labels: np.ndarray
        The predicted labels to evaluate.
    metric: str or list of str
        The metric(s) to use for evaluation. Can be a single metric or a list of metrics. Supported metrics are defined in supervised_metric_list() and unsupervised_metric_list().

    Returns:
    --------
    dict or float
        If metric is a list, returns a dictionary with metric names as keys and their corresponding scores as values. If metric is a single string, returns the score for that metric.
    """
    if type(metric) == list:
        scores = {}
        for m in metric:
            if m in supervised_metric_list():
                scores[m] = metric_functions(m)(true_labels, pred_labels)
            elif m in unsupervised_metric_list():
                eval = [metric_functions(m)(x, pred_labels) for x in X_views]
                scores[m] = np.mean(eval)
        return scores
    # If metric is a single string, return the score directly.
    if metric in supervised_metric_list():
        return metric_functions(metric)(true_labels, pred_labels)
    elif metric in unsupervised_metric_list():
        eval = [metric_functions(metric)(x, pred_labels) for x in X_views]
        return np.mean(eval)

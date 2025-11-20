from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import numpy as np


def metric_functions(metric: str):
    """ """
    metrics_fun = {
        "chs": calinski_harabasz_score,
        "sil": silhouette_score,
        "ari": adjusted_rand_score,
        "nmi": normalized_mutual_info_score,
        "ami": adjusted_mutual_info_score,
        "acc": acc,
        "dbs": davies_bouldin_score,
    }
    return metrics_fun[metric]


def supervised_metric_list():
    """
    Returns the list of available supervised metric
    """
    return ["ami", "nmi", "ari", "acc", "pts"]


def unsupervised_metric_list():
    """
    Returns the list of available unsupervised metric
    """
    return ["sil", "dbs", "chs", "sch"]


def pts(y_true, y_pred):
    """Purity score
    Args:
      y_true(np.ndarray): n*1 matrix Ground truth labels
      y_pred(np.ndarray): n*1 matrix Predicted clusters

    Returns:
      float: Purity score
    """
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return acc(y_true, y_voted_labels)

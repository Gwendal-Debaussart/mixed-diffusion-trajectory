import numpy as np
import json
import os
from .helix_a import helix_a
from .helix_b import helix_b
from .deformed_plane import deformed_plane
from .mnist_lindenbaum import mnist_lindenbaum
from .isolet_lindenbaum import isolet_lindenbaum
from .mnist_kuchroo import mnist_kuchroo
from .multiple_feat import multiple_feat
from .olivetti import olivetti
from .caltech101_7 import caltech101_7
from .leaves import leaves
from .msrc import msrc
from .yale import yale

from .utilities import get_kernel_matrix


def load_dataset(name, **args):
    """
    Load a dataset by name.

    Parameters:
    -----------
    name : str
        Name of the dataset to load.
    **args : dict
        Additional arguments to pass to the dataset loading function.

    Returns:
    --------
    data : tuple
        The loaded dataset, typically as (X, Y) or (list_views, true_labels).
    """
    datasets = {
        "helix_a": helix_a,
        "helix_b": helix_b,
        "deformed_plane": deformed_plane,
        "mnist_lindenbaum": mnist_lindenbaum,
        "isolet_lindenbaum": isolet_lindenbaum,
        "mnist_kuchroo": mnist_kuchroo,
        "multiple_feat": multiple_feat,
        "olivetti": olivetti,
        "caltech101-7": caltech101_7,
        "leaves": leaves,
        "msrc": msrc,
        "yale": yale,
    }

    if name in datasets:
        return datasets[name](**args)
    else:
        raise ValueError(f"Dataset '{name}' is not recognized.")


def load_preprocessed_dataset(name, return_views=False, **args):
    """
    Load and preprocess a dataset by name.

    Parameters:
    -----------
    name : str
        Name of the dataset to load.
    **args : dict
        Additional arguments to pass to the dataset loading function.

    Returns:
    --------
    data : tuple
        The preprocessed dataset, typically as (X_preprocessed, Y).
    """
    X, Y = load_dataset(name, **args)
    X_preprocessed = []
    if name not in ["isolet_lindenbaum"]:
        for x in X:
            K = get_kernel_matrix(x, normalize=True)
            X_preprocessed.append(K)
    else:
        X_preprocessed = []
        for x in X:
            row_sums = x.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_preprocessed.append(x / row_sums)
    if return_views:
        return X_preprocessed, X, Y
    return X_preprocessed, Y


def get_num_clusters(dataset_name):
    """
    Get or compute the number of clusters for a given dataset.

    Reads from ../tables/dataset_infos/dataset_clusters.json relative to this file.
    If missing, computes from labels (len(unique(Y))) and updates the JSON.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        int: Number of clusters in the dataset.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "tables", "dataset_infos")
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "dataset_clusters.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                dataset_clusters = json.load(f)
            except json.JSONDecodeError:
                print(f"[Warning] Corrupted JSON file, reinitializing {json_path}")
                dataset_clusters = {}
    else:
        dataset_clusters = {}

    if dataset_name in dataset_clusters:
        return dataset_clusters[dataset_name]

    try:
        _, Y = load_preprocessed_dataset(dataset_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")

    num_clusters = int(len(np.unique(Y)))
    dataset_clusters[dataset_name] = num_clusters

    with open(json_path, "w") as f:
        json.dump(dataset_clusters, f, indent=4)

    return num_clusters

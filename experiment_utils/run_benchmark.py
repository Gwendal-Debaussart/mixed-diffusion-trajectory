from .get_operator_from_method import get_operator_from_method
from .method_to_embedding import method_to_embedding
from utilities.evaluate import get_clustering, evaluate_labels
from .save_results import save_raw_results
import numpy as np
import pandas as pd
from benchmarks import load_preprocessed_dataset
import os
from joblib import Parallel, delayed


def run_one_repeat(
    method, dataset_name, num_clusters, dim_embedd, X_preprocessed, X_views, Y
):
    """ """
    operator = get_operator_from_method(method, dataset_name, X_preprocessed, X_views)
    embedding = method_to_embedding(operator, X_views, method, dim_embedd)

    # Truncate the embedding if MVDM, keep the first n_samples rows
    n_samples = len(X_views[0])
    if isinstance(embedding, np.ndarray) and embedding.shape[0] > n_samples:
        embedding = embedding[:n_samples, :]

    labels = get_clustering(embedding, num_clusters)
    score = evaluate_labels(Y, X_preprocessed, labels, metric=["chs", "ami", "ari"])
    return score


def get_existing_repeats(
    dataset, method_name, save_dir="tables/clustering_benchmark_raw/"
):
    """
    Get the number of repeats already computed for a method (and view if single-view).

    Returns:
        int: number of repeats already computed.
    """
    fname = dataset["name"]
    if "noise_factor" in dataset:
        fname += f"_noise_{dataset['noise_factor']:.2f}"
    filepath = os.path.join(save_dir, f"{fname}.csv")

    if not os.path.exists(filepath):
        return 0

    df = pd.read_csv(filepath)
    existing = df[df["method"] == method_name]
    if existing.empty:
        return 0
    return existing["repeat"].max() + 1


def run_benchmark_for_dataset(
    dataset, methods, num_repeats=10, save_dir="tables/clustering_benchmark_raw/"
):
    """
    Run benchmark for one dataset.
    """
    # ------------------- Load dataset
    dataset_name = dataset["name"]
    noise_factor = dataset.get("noise_factor", None)
    try:
        if noise_factor is not None:
            X_preprocessed, X_views, Y = load_preprocessed_dataset(
                dataset_name, return_views=True, noise_factor=noise_factor
            )
        else:
            X_preprocessed, X_views, Y = load_preprocessed_dataset(
                dataset_name, return_views=True
            )
    except TypeError:
        X_preprocessed, X_views, Y = load_preprocessed_dataset(
            dataset_name, return_views=True
        )
    num_clusters = len(np.unique(Y))
    dim_embedd = num_clusters
    all_results = []

    for method in methods:
        if "n_views" in method and method["n_views"] < len(X_views):
            print(
                f"Skipping {method['name']} for dataset {dataset_name} ({len(X_views)} views)"
            )
            continue
        if method.get("single_view", False):
            for view_idx in range(len(X_views)):
                method_copy = method.copy()
                method_copy["name"] = f"{method['name']} (view {view_idx+1})"
                existing_repeats = get_existing_repeats(
                    dataset, method_name=method_copy["name"]
                )
                repeats_needed = num_repeats - existing_repeats
                if repeats_needed == 0:
                    print(
                        f"[{dataset_name}] (nf = {noise_factor}) already computed [{method_copy['name']}]"
                    )
                    continue
                if method.get("stochastic", False):
                    repeats = Parallel(n_jobs=-1)(
                        delayed(run_one_repeat)(
                            method=method_copy,
                            dataset_name=dataset_name,
                            num_clusters=num_clusters,
                            dim_embedd=dim_embedd,
                            X_preprocessed=[X_preprocessed[view_idx]],
                            X_views=[X_views[view_idx]],
                            Y=Y,
                        )
                        for _ in range(repeats_needed)
                    )
                else:
                    operator = get_operator_from_method(
                        method_copy,
                        dataset_name,
                        [X_preprocessed[view_idx]],
                        [X_views[view_idx]],
                    )
                    embedding = method_to_embedding(
                        operator, [X_views[view_idx]], method_copy, dim_embedd
                    )
                    n_samples = len(X_views[view_idx])
                    repeats = [
                        evaluate_labels(
                            true_labels=Y,
                            Xv=X_preprocessed,
                            pred_labels=get_clustering(embedding, num_clusters),
                            metric=["chs", "ami", "ari"],
                        )
                        for _ in range(repeats_needed)
                    ]
                print(
                    f"[{dataset_name}] (n.f = {noise_factor}) Completed [{repeats_needed}] repeats for {method['name']}"
                )
                save_raw_results(dataset, method_copy["name"], repeats)
                all_results.extend(repeats)
            continue
        existing_repeats = get_existing_repeats(dataset, method_name=method["name"])
        repeats_needed = num_repeats - existing_repeats
        if repeats_needed == 0:
            print(f"[{dataset_name}] {method['name']}: already has {num_repeats} runs")
            continue

        print(
            f"[{dataset_name}] Running {method['name']} ({repeats_needed} missing repeats)"
        )
        if method.get("stochastic", False):
            repeats = Parallel(n_jobs=-1)(
                delayed(run_one_repeat)(
                    method=method,
                    dataset_name=dataset_name,
                    num_clusters=num_clusters,
                    dim_embedd=dim_embedd,
                    X_preprocessed=X_preprocessed,
                    X_views=X_views,
                    Y=Y,
                )
                for _ in range(repeats_needed)
            )
        else:
            operator = get_operator_from_method(
                method, dataset_name, X_preprocessed, X_views
            )
            embedding = method_to_embedding(operator, X_views, method, dim_embedd)
            n_samples = len(X_views[0]) if X_views else X_preprocessed.shape[0]
            if embedding.shape[0] > n_samples:
                embedding = embedding[:n_samples, :n_samples]

            repeats = [
                evaluate_labels(
                    true_labels=Y,
                    Xv=X_preprocessed,
                    pred_labels=get_clustering(embedding, num_clusters),
                    metric=["chs", "ami", "ari"],
                )
                for _ in range(repeats_needed)
            ]
        print(
            f"[{dataset_name}] Completed [{repeats_needed}] repeats for {method['name']}"
        )
        save_raw_results(dataset, method["name"], repeats, save_dir=save_dir)
        all_results.extend(repeats)
    return all_results

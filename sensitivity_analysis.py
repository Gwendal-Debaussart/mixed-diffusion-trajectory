from mdt.random_mdt import random_mdt_operator
from mdt.mdt_direct import mdt_direct
from benchmarks import load_preprocessed_dataset
import pandas as pd
import os
from joblib import Parallel, delayed
from experiments.run_benchmark import run_one_repeat, get_existing_repeats
from experiments.method_to_embedding import method_to_embedding
from experiments.get_operator_from_method import get_operator_from_method
from utilities.evaluate import evaluate_labels, get_clustering
from benchmarks.load_dataset import get_num_clusters
from experiments.save_results import save_raw_results
import multiprocessing


def run_t_sensitivity(
    dataset,
    methods,
    t_values=range(1, 21),
    num_repeats=100,
    save_dir="tables/sensitivity_analysis/",
):
    """
    Run sensitivity analysis for diffusion time parameter t.
    Reuses the same evaluation + saving pipeline as the benchmark.

    Args:
        dataset (dict): dataset config (with 'name', 'noise_factor', etc.)
        methods (list): list of method dicts (see examples above)
        t_values (iterable): list or range of t values to test
        num_repeats (int): number of repeats per t
        save_dir (str): base directory for saving results
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = dataset["name"]
    noise_factor = dataset.get("noise_factor", None)
    try:
        X_preprocessed, X_views, Y = load_preprocessed_dataset(
            dataset_name,
            return_views=True,
            noise_factor=noise_factor if noise_factor is not None else None,
        )
    except TypeError:
        X_preprocessed, X_views, Y = load_preprocessed_dataset(
            dataset_name, return_views=True
        )

    all_results = []

    num_clusters = get_num_clusters(dataset_name)
    dim_embedd = num_clusters

    for method in methods:
        print(f"\n[{dataset_name}] Testing {method['name']} across t values")

        for t in t_values:
            method_t = method.copy()
            method_t["params"] = lambda dn, t=t: {
                **method["params"](dn),
                "t": t,
            }
            method_t["name"] = f"{method['name']}_t{t}"
            existing_repeats = get_existing_repeats(
                dataset, method_name=method_t["name"], save_dir=save_dir
            )
            if existing_repeats >= num_repeats:
                print(
                    f"  -> {method_t['name']} already has {existing_repeats} repeats, skipping"
                )
                continue
            repeats_needed = num_repeats - existing_repeats
            if method.get("stochastic", False):
                repeats = Parallel(n_jobs=-1)(
                    delayed(run_one_repeat)(
                        method=method_t,
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
                # Deterministic method – only one computation
                operator = get_operator_from_method(
                    method_t, dataset_name, X_preprocessed, X_views
                )
                embedding = method_to_embedding(operator, X_views, method_t, dim_embedd)
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

            save_raw_results(dataset, method_t["name"], repeats, save_dir=save_dir)
            all_results.extend(repeats)

            print(f"  -> {method['name']} t={t} done")

    return all_results


methods = (
    {
        "name": "Random Convex MDT",
        "func": random_mdt_operator,
        "input_type": "preprocessed",
        "decomp_method": "svd",
        "stochastic": True,
        "params": lambda dn: {"t": None, "convex": True},
    },
    {
        "name": "Random MDT",
        "func": random_mdt_operator,
        "input_type": "preprocessed",
        "decomp_method": "svd",
        "stochastic": True,
        "params": lambda dn: {"t": None, "convex": False},
    },
    # {
    #     "name": "Direct MDT",
    #     "func": mdt_direct,
    #     "input_type": "preprocessed",
    #     "decomp_method": "svd",
    #     "params": lambda dn: {"t": None, "k": get_num_clusters(dn)},
    # },
)

datasets = [
    {"name": "isolet_lindenbaum"},
    {"name": "multiple_feat"},
    {"name": "olivetti"},
    {"name": "caltech101-7"},
    {"name": "leaves"},
    {"name": "yale"},
    {"name": "msrc"},
    {"name": "mnist_lindenbaum", "noise_factor": 0.5},
    {"name": "mnist_kuchroo", "noise_factor": 0.5},
]
n_cores = multiprocessing.cpu_count()
n_jobs = max(8, n_cores // 4)

Parallel(n_jobs=n_jobs)(
    delayed(run_t_sensitivity)(dataset, methods, t_values=list(range(1, 26)) + list(range(30, 55, 5)), num_repeats=100)
    for dataset in datasets
)

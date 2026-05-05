import os
import sys

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from benchmarks.load_dataset import get_num_clusters, load_preprocessed_dataset
from competitors.gcca import gcca_embedding
from competitors.mvsc import mvsc_embedding
from experiment_utils.get_diffusion_time import get_diffusion_time
from experiment_utils.save_results import format_results, save_raw_results
from mdt.random_mdt import random_mdt_operator
from utilities.evaluate import evaluate_labels, get_clustering


COMPUTE_MNIST_DATASETS = False
COMPUTE_PARTIAL_MNIST_DATASETS = True


def method_list():
    return (
        {
            "name": "GCCA + MDT",
            "func": gcca_embedding,
            "params": lambda dataset: {
                "n_components": get_num_clusters(dataset["name"]),
                "diffusion_time": get_diffusion_time(
                    dataset["name"],
                    **({"noise_factor": dataset["noise_factor"]} if "noise_factor" in dataset else {}),
                ),
            },
            "stochastic": True,
            "n_trajectories": 10,
        },
        {
            "name": "MVSC + MDT",
            "func": mvsc_embedding,
            "params": lambda dataset: {
                "n_clusters": get_num_clusters(dataset["name"]),
                "diffusion_time": get_diffusion_time(
                    dataset["name"],
                    **({"noise_factor": dataset["noise_factor"]} if "noise_factor" in dataset else {}),
                ),
            },
            "stochastic": True,
            "n_trajectories": 10,
        },
    )


def precompute_diffusion_times(datasets, n_jobs=-1):
    def compute_for_dataset(dataset):
        kwargs = {}
        if "noise_factor" in dataset:
            kwargs["noise_factor"] = dataset["noise_factor"]
        t = get_diffusion_time(dataset["name"], **kwargs)
        print(f"[{dataset['name']}] diffusion time computed: t={t}")
        return t

    Parallel(n_jobs=n_jobs)(delayed(compute_for_dataset)(dataset) for dataset in datasets)


def get_existing_repeats(dataset, method_name, save_dir="tables/clustering_benchmark_raw/"):
    fname = dataset["name"]
    if "noise_factor" in dataset:
        fname += f"_noise_{dataset['noise_factor']:.2f}"
    filepath = os.path.join(save_dir, f"{fname}.csv")

    if not os.path.exists(filepath):
        return 0

    import pandas as pd

    df = pd.read_csv(filepath)
    existing = df[df["method"] == method_name]
    if existing.empty:
        return 0
    return existing["repeat"].max() + 1


def load_dataset_with_views(dataset_name, noise_factor=None):
    if noise_factor is not None:
        loaded = load_preprocessed_dataset(
            dataset_name, return_views=True, noise_factor=noise_factor
        )
    else:
        loaded = load_preprocessed_dataset(dataset_name, return_views=True)

    assert len(loaded) == 3
    X_preprocessed, X_views, Y = loaded
    return X_preprocessed, X_views, Y


def build_shared_trajectories(X_preprocessed, diffusion_time, n_trajectories):
    return [
        random_mdt_operator(
            X_preprocessed,
            diffusion_time,
            convex=True,
            distribution="dirichlet",
        )
        for _ in range(n_trajectories)
    ]


def score_method_on_trajectories(method, dataset, trajectories, num_clusters, X_preprocessed, Y):
    params = method.get("params", lambda ds: {})(dataset)
    embedding = method["func"](trajectories, **params)
    labels = get_clustering(embedding, num_clusters)
    return evaluate_labels(Y, X_preprocessed, labels, metric=["chs", "ami", "ari"])


def run_benchmark_for_dataset(dataset, methods, num_repeats=50, save_dir="tables/clustering_benchmark_raw/"):
    dataset_name = dataset["name"]
    noise_factor = dataset.get("noise_factor", None)

    X_preprocessed, X_views, Y = load_dataset_with_views(dataset_name, noise_factor=noise_factor)

    num_clusters = len(np.unique(Y))
    method_states = []
    max_repeats_needed = 0

    for method in methods:
        existing_repeats = get_existing_repeats(dataset, method_name=method["name"], save_dir=save_dir)
        repeats_needed = num_repeats - existing_repeats
        method_states.append(
            {
                "method": method,
                "existing_repeats": existing_repeats,
                "repeats_needed": repeats_needed,
            }
        )
        max_repeats_needed = max(max_repeats_needed, repeats_needed)

    if max_repeats_needed <= 0:
        for method_state in method_states:
            print(
                f"[{dataset_name}] (nf = {noise_factor}) {method_state['method']['name']}: already has {num_repeats} runs"
            )
        return []

    n_trajectories = max(method.get("n_trajectories", 10) for method in methods)
    shared_results = []

    print(
        f"[{dataset_name}] (n.f = {noise_factor}) Running shared MDT trajectories for {max_repeats_needed} repeats"
    )
    for _ in range(max_repeats_needed):
        trajectories = build_shared_trajectories(X_preprocessed, diffusion_time=method_states[0]["method"]["params"](dataset)["diffusion_time"], n_trajectories=n_trajectories)
        repeat_results = {}
        for method_state in method_states:
            method = method_state["method"]
            if method_state["repeats_needed"] <= 0:
                continue
            repeat_results[method["name"]] = score_method_on_trajectories(
                method=method,
                dataset=dataset,
                trajectories=trajectories,
                num_clusters=num_clusters,
                X_preprocessed=X_preprocessed,
                Y=Y,
            )
        shared_results.append(repeat_results)

    for method_state in method_states:
        method = method_state["method"]
        repeats_needed = method_state["repeats_needed"]
        if repeats_needed <= 0:
            continue
        repeats = [repeat_results[method["name"]] for repeat_results in shared_results if method["name"] in repeat_results][:repeats_needed]
        print(f"[{dataset_name}] Completed [{len(repeats)}] repeats for {method['name']}")
        save_raw_results(dataset, method["name"], repeats, save_dir=save_dir)

    return shared_results


def run_benchmark(datasets, methods, num_repeats=50, n_jobs=2, save_dir="tables/clustering_benchmark_raw/"):
    precompute_diffusion_times(datasets)
    Parallel(n_jobs=n_jobs)(
        delayed(run_benchmark_for_dataset)(dataset, methods, num_repeats, save_dir)
        for dataset in datasets
    )


if __name__ == "__main__":
    methods = method_list()

    datasets = [
        {"name": "isolet_lindenbaum"},
        {"name": "multiple_feat"},
        {"name": "olivetti"},
        {"name": "caltech101-7"},
        {"name": "leaves"},
        {"name": "yale"},
        {"name": "msrc"},
    ]

    if COMPUTE_PARTIAL_MNIST_DATASETS:
        mnist_noise_datasets = [
            {"name": f"mnist_{name}", "noise_factor": float(0.5)}
            for name in ["lindenbaum", "kuchroo"]
        ]
        datasets.extend(mnist_noise_datasets)

    if COMPUTE_MNIST_DATASETS:
        noise_values = np.arange(0.05, 1, 0.05)
        mnist_noise_datasets = [
            {"name": f"mnist_{name}", "noise_factor": float(s)}
            for s in noise_values
            for name in ["lindenbaum", "kuchroo"]
        ]
        datasets.extend(mnist_noise_datasets)

    n_cores = multiprocessing.cpu_count()
    n_jobs = 2

    run_benchmark(
        datasets,
        methods,
        num_repeats=50,
        n_jobs=n_jobs,
        save_dir="tables/clustering_benchmark_raw/",
    )

    for dataset in datasets:
        format_results(dataset["name"], dataset.get("noise_factor", None))

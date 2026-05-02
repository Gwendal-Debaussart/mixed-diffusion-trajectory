import os
import sys

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from utilities import *
from mdt import *
from competitors import *
from experiment_utils.run_benchmark import run_benchmark_for_dataset
from experiment_utils.method_list import method_list
from experiment_utils import *
from joblib import Parallel, delayed
import multiprocessing

# Options to control which datasets to run, avoid having both to True.

COMPUTE_MNIST_DATASETS = False  # Set to True to run the benchmark on the mnist-based datasets with noise (takes much longer)

COMPUTE_PARTIAL_MNIST_DATASETS = False  # Set to True to run the benchmark on the mnist-based datasets with noise_factor=0.5 only (faster, for testing)


"""
Method are represented as dictionaries with keys:
  'name': str name of the method.
  'type': 'views' | 'single_view' | 'default' (others)
  'func': callable to compute the operator.
  'params': callable that takes dataset_name and returns a dict of parameters. e.g to get the diffusion time of the dataset.
  'stochastic': bool indicating if the operator is stochastic (for evaluation purposes).

  see method_list() for examples.
"""

"""
This file contains the benchmark for clustering methods on the 'larger' multi-view datasets. (Mnist-based datasets).
"""

def precompute_diffusion_times(datasets, n_jobs=-1):
    """
    Compute diffusion time for all datasets in parallel, ensuring the diffusion_times.csv
    file is populated before running the benchmark.

    Args:
        datasets (list of dict): Each dict should contain at least 'name' and optional 'noise_factor'.
        n_jobs (int): Number of parallel jobs for joblib. Default -1 uses all cores.
    """

    def compute_for_dataset(dataset):
        name = dataset["name"]
        kwargs = {}
        if "noise_factor" in dataset:
            kwargs["noise_factor"] = dataset["noise_factor"]
        t = get_diffusion_time(name, **kwargs)
        print(f"[{name}] diffusion time computed: t={t}")
        return t

    Parallel(n_jobs=n_jobs)(
        delayed(compute_for_dataset)(dataset) for dataset in datasets
    )


def run_benchmark(
    datasets,
    methods,
    num_repeats=100,
    n_jobs=-1,
    save_dir="tables/clustering_benchmark_raw/",
):
    """
    Run benchmark for multiple datasets and methods.
    """
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

    # n_jobs = max(4, n_cores // 4)
    n_jobs = 3

    run_benchmark(
        datasets,
        methods,
        num_repeats=100,
        n_jobs=n_jobs,
        save_dir="tables/clustering_benchmark_raw/",
    )
    for dataset in datasets:
        format_results(dataset["name"], dataset.get("noise_factor", None))

"""
Benchmark script to analyze runtime scaling of MDT computation steps with data size.

This script measures how each component of the MDT pipeline scales as the number
of data points increases:
- Time selection (finding optimal diffusion time)
- Operator computation (product of weighted kernel matrices)
- Embedding step (SVD/eigendecomposition)
- Clustering step (K-means)

Uses random Gaussian data for testing.
"""

import os
import sys

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt
from kneed import KneeLocator

from benchmarks.utilities import get_kernel_matrix
from utilities.evaluate import get_embedding, get_clustering
from utilities.entropy import entropy_from_values
from mdt.mdt_utils import mdt_operator
from experiment_utils.get_diffusion_time import get_diffusion_time
from visualization.style import get_col_list


# Color mapping for pipeline steps (ordered by user preference)
_STEP_NAMES = [
    "operator_computation",
    "embedding_computation",
    "kernel_computation",
    "clustering",
    "trajectory_generation",
    "diffusion_entropy_and_elbow",
]
_PALETTE = get_col_list()
STEP_COLORS = {
    step: _PALETTE[i % len(_PALETTE)]
    for i, step in enumerate(_STEP_NAMES)
}
N_SAMPLES_LIST = [100, 200, 500, 1000, 1500, 2000, 3000, 4000]


def generate_multiview_gaussian(n_samples, n_views=2, n_features_per_view=10, seed=None):
    """
    Generate random multi-view Gaussian data.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_views : int
        Number of views
    n_features_per_view : int
        Features per view
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    X_views : list of np.ndarray
        List of feature matrices (one per view)
    """
    if seed is not None:
        np.random.seed(seed)

    X_views = [
        np.random.randn(n_samples, n_features_per_view)
        for _ in range(n_views)
    ]

    # Standardize
    X_views = [StandardScaler().fit_transform(X) for X in X_views]
    return X_views


def time_kernel_computation(X_views):
    """
    Measure time to compute kernel matrices from views.

    Parameters:
    -----------
    X_views : list of np.ndarray
        Feature matrices

    Returns:
    --------
    tuple : (time_in_seconds, kernel_matrices)
    """
    start = time.time()
    X_kernels = [get_kernel_matrix(X, normalize=True) for X in X_views]
    elapsed = time.time() - start
    return elapsed, X_kernels


def time_trajectory_generation(X_kernels, n_timepoints=5):
    """
    Measure time to generate random diffusion trajectory.

    Parameters:
    -----------
    X_kernels : list of np.ndarray
        Kernel matrices
    n_timepoints : int
        Number of time steps in trajectory

    Returns:
    --------
    tuple : (time_in_seconds, trajectory)
    """
    n_views = len(X_kernels)

    start = time.time()
    # Generate random trajectory (normalized weights)
    trajectory = np.random.rand(n_timepoints, n_views)
    trajectory = trajectory / trajectory.sum(axis=1, keepdims=True)
    elapsed = time.time() - start

    return elapsed, trajectory


def time_operator_computation(trajectory, X_kernels):
    """
    Measure time to compute the MDT operator.

    Parameters:
    -----------
    trajectory : np.ndarray
        Weight trajectory of shape (n_timepoints, n_views)
    X_kernels : list of np.ndarray
        Kernel matrices

    Returns:
    --------
    tuple : (time_in_seconds, operator)
    """
    start = time.time()
    operator = mdt_operator(trajectory, X_kernels)
    elapsed = time.time() - start

    return elapsed, operator


def time_embedding_computation(operator, n_components=10, method="svd"):
    """
    Measure time to compute embedding from operator.

    Parameters:
    -----------
    operator : np.ndarray
        Operator matrix
    n_components : int
        Number of embedding components
    method : str
        Embedding method ('svd', 'eigen', etc.)

    Returns:
    --------
    tuple : (time_in_seconds, embedding)
    """
    start = time.time()
    embedding = get_embedding(operator, n_components, method=method)
    elapsed = time.time() - start

    return elapsed, embedding


def time_clustering(embedding, n_clusters=3):
    """
    Measure time to perform K-means clustering.

    Parameters:
    -----------
    embedding : np.ndarray
        Embedding matrix
    n_clusters : int
        Number of clusters

    Returns:
    --------
    tuple : (time_in_seconds, labels)
    """
    start = time.time()
    labels = get_clustering(embedding, n_clusters)
    elapsed = time.time() - start

    return elapsed, labels


def time_diffusion_entropy_and_elbow(X_kernels, t_max=25):
    """
    Measure time to compute diffusion entropy across timepoints and find elbow.

    Parameters:
    -----------
    X_kernels : list of np.ndarray
        Kernel matrices
    t_max : int
        Maximum diffusion time to evaluate

    Returns:
    --------
    tuple : (time_in_seconds, optimal_t, entropies)
    """

    start = time.time()
    operator = np.mean(X_kernels, axis=0)
    running_operator = np.eye(operator.shape[0]) # Start with identity for t=0
    entropies = []
    for t in range(1, t_max + 1):
        # Generate trajectory for this time step (uniform weights)
        running_operator = running_operator @ operator
        singular_vals = np.linalg.svd(running_operator, compute_uv=False)
        entropy_val = entropy_from_values(singular_vals)
        entropies.append(entropy_val)

    # Find elbow point using KneeLocator
    entropies = np.array(entropies)
    knee_locator = KneeLocator(
        range(1, len(entropies) + 1),
        entropies,
        curve="convex",
        direction="decreasing"
    )
    optimal_t = knee_locator.knee if knee_locator.knee is not None else 1

    elapsed = time.time() - start

    return elapsed, optimal_t, entropies


def benchmark_mdt_pipeline(n_samples_list, n_views=2, n_features=10, n_clusters=3,
                           n_repeats=3, seed=42):
    """
    Benchmark the MDT pipeline across different data sizes.

    Parameters:
    -----------
    n_samples_list : list of int
        List of sample sizes to test
    n_views : int
        Number of views
    n_features : int
        Features per view
    n_clusters : int
        Number of clusters for benchmarking
    n_repeats : int
        Number of repeats per configuration
    seed : int
        Random seed

    Returns:
    --------
    pd.DataFrame
        Results with columns: n_samples, step, time_mean, time_std
    """

    results = []

    for n_samples in n_samples_list:
        print(f"\nBenchmarking n_samples = {n_samples}")

        step_times = {
            "kernel_computation": [],
            "diffusion_entropy_and_elbow": [],
            "trajectory_generation": [],
            "operator_computation": [],
            "embedding_computation": [],
            "clustering": [],
            "total_pipeline": [],
        }

        for repeat in range(n_repeats):
            print(f"  Repeat {repeat + 1}/{n_repeats}", end="\r")

            # Generate data
            X_views = generate_multiview_gaussian(
                n_samples, n_views=n_views, n_features_per_view=n_features,
                seed=seed + repeat
            )

            # Time each component
            t_kernel, X_kernels = time_kernel_computation(X_views)
            step_times["kernel_computation"].append(t_kernel)

            t_entropy, optimal_t, entropies = time_diffusion_entropy_and_elbow(X_kernels, t_max=50)
            step_times["diffusion_entropy_and_elbow"].append(t_entropy)

            t_traj, trajectory = time_trajectory_generation(X_kernels, n_timepoints=5)
            step_times["trajectory_generation"].append(t_traj)

            t_op, operator = time_operator_computation(trajectory, X_kernels)
            step_times["operator_computation"].append(t_op)

            t_emb, embedding = time_embedding_computation(operator, n_components=n_clusters)
            step_times["embedding_computation"].append(t_emb)

            t_clust, labels = time_clustering(embedding, n_clusters=n_clusters)
            step_times["clustering"].append(t_clust)

            # Total time
            t_total = t_kernel + t_entropy + t_traj + t_op + t_emb + t_clust
            step_times["total_pipeline"].append(t_total)

        print(f"  Repeat {n_repeats}/{n_repeats} ✓")

        # Aggregate results
        for step, times in step_times.items():
            results.append({
                "n_samples": n_samples,
                "step": step,
                "time_mean": np.mean(times),
                "time_std": np.std(times),
                "times": times,
            })

    return pd.DataFrame(results)


def plot_results(results, save_path="figures/runtime_scaling/"):
    """
    Plot runtime scaling results as stacked area plots saved as separate PDFs.
    Creates two plots: one for elbow detection, one for other pipeline steps.

    Parameters:
    -----------
    results : pd.DataFrame
        Results dataframe
    save_path : str
        Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    all_steps = [s for s in results["step"].unique() if s != "total_pipeline" and "top_k" not in s]
    n_samples_list = sorted(results["n_samples"].unique())
    x = np.array(n_samples_list)

    elbow_steps = [s for s in all_steps if "elbow" in s or "entropy" in s]
    other_steps = [s for s in all_steps if s not in elbow_steps]

    step_order = [
        "operator_computation",
        "embedding_computation",
        "kernel_computation",
        "clustering",
        "trajectory_generation",
    ]

    if elbow_steps:
        fig, ax = plt.subplots()
        data_dict = {}
        for step in elbow_steps:
            step_data = results[results["step"] == step].sort_values("n_samples")
            data_dict[step] = step_data["time_mean"].values

        sorted_elbow = sorted(elbow_steps)
        y_data = [data_dict[step] for step in sorted_elbow]
        labels = [step.replace("_", " ").title() for step in sorted_elbow]
        colors = [STEP_COLORS.get(step, "#CCCCCC") for step in sorted_elbow]

        ax.stackplot(
            x,
            *reversed(y_data),
            labels=list(reversed(labels)),
            colors=list(reversed(colors)),
            alpha=1,
            edgecolor="white",
            linewidth=0.2,
        )
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Time (seconds)")
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="best", fontsize=14, reverse=True)
        ax.grid(alpha=0.25)
        ax.set_xlim(n_samples_list[0], n_samples_list[-1])

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "mdt_runtime_scaling_time_selection.pdf"))
        print(f"Plot saved to {save_path}mdt_runtime_scaling_time_selection.pdf")
        plt.close()

    if other_steps:
        fig, ax = plt.subplots()
        data_dict = {}
        for step in other_steps:
            step_data = results[results["step"] == step].sort_values("n_samples")
            data_dict[step] = step_data["time_mean"].values

        sorted_other = [s for s in step_order if s in other_steps]
        y_data = [data_dict[step] for step in sorted_other]
        labels = [step.replace("_", " ").title() for step in sorted_other]
        colors = [STEP_COLORS.get(step, "#CCCCCC") for step in sorted_other]

        ax.stackplot(
            x,
            *reversed(y_data),
            labels=list(reversed(labels)),
            colors=list(reversed(colors)),
            alpha=1,
            edgecolor="white",
            linewidth=0.2,
        )
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("Time (seconds)")
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="best", fontsize=14, reverse=True)
        ax.grid(alpha=0.25)
        ax.set_xlim(n_samples_list[0], n_samples_list[-1])

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "mdt_runtime_scaling_pipeline.pdf"))
        print(f"Plot saved to {save_path}mdt_runtime_scaling_pipeline.pdf")
        plt.close()


def load_results_csv(save_path="tables/runtime_scaling/"):
    """
    Load results from CSV if it exists.

    Parameters:
    -----------
    save_path : str
        Directory to load CSV from

    Returns:
    --------
    pd.DataFrame or None
        Results dataframe if file exists, None otherwise
    """
    filepath = os.path.join(save_path, "mdt_runtime_N.csv")
    if os.path.exists(filepath):
        print(f"Loading existing results from {filepath}")
        return pd.read_csv(filepath)
    return None


def save_results_csv(results, save_path="tables/runtime_scaling/"):
    """
    Save results to CSV.

    Parameters:
    -----------
    results : pd.DataFrame
        Results dataframe
    save_path : str
        Directory to save CSV
    """
    os.makedirs(save_path, exist_ok=True)

    # Remove the 'times' column for CSV (keep only aggregates)
    results_to_save = results.drop(columns=["times"])

    filepath = os.path.join(save_path, "mdt_runtime_N.csv")
    results_to_save.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def print_summary(results):
    """
    Print summary statistics.

    Parameters:
    -----------
    results : pd.DataFrame
        Results dataframe
    """
    print("\n" + "="*70)
    print("MDT PIPELINE RUNTIME SCALING SUMMARY")
    print("="*70)

    for step in sorted(results["step"].unique()):
        step_data = results[results["step"] == step].sort_values("n_samples")
        print(f"\n{step.replace('_', ' ').upper()}")
        print("-" * 70)
        print(f"{'N_samples':<12} {'Mean Time (s)':<18} {'Std Dev':<15}")
        print("-" * 70)
        for _, row in step_data.iterrows():
            print(f"{int(row['n_samples']):<12} {row['time_mean']:<18.6f} {row['time_std']:<15.6f}")


if __name__ == "__main__":
    # Configuration
    n_samples_list = N_SAMPLES_LIST
    n_views = 2
    n_features = 10
    n_clusters = 3
    n_repeats = 3

    print("MDT Runtime Scaling Benchmark")
    print("=" * 70)
    print(f"N_samples: {n_samples_list}")
    print(f"N_views: {n_views}")
    print(f"N_features_per_view: {n_features}")
    print(f"N_clusters: {n_clusters}")
    print(f"N_repeats: {n_repeats}")
    print("=" * 70)

    # Try to load existing results
    results = load_results_csv()

    if results is None:
        print("\nRunning new benchmark...\n")
        # Run benchmark
        results = benchmark_mdt_pipeline(
            n_samples_list=n_samples_list,
            n_views=n_views,
            n_features=n_features,
            n_clusters=n_clusters,
            n_repeats=n_repeats,
        )
        # Save results
        save_results_csv(results)
    else:
        print("\nUsing cached results. Delete CSV to rerun benchmark.\n")

    # Print summary
    print_summary(results)

    # Plot results
    plot_results(results)

    print("\n✓ Benchmark complete!")

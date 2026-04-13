import os
import time
import sys

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from competitors.alternating_diffusion import alternating_diffusion
from competitors.integrated_diffusion import integrated_diffusion
from competitors.multiview_diffusion import multiview_diffusion
from mdt.mdt_direct import mdt_direct
from mdt.random_mdt import random_mdt_operator
from mdt.mdt_contrastive import mdt_contrastive
from experiment_utils import get_diffusion_time
from benchmarks.utilities import get_kernel_matrix
from experiment_utils.method_to_embedding import get_embedding

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SIZES           = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
REPEATS         = 3
DIRECT_REPEATS  = 1
TRAJECTORY_T    = 10
SELECTION_MAX_T = 25
CENTERS         = 4
CLUSTER_STD     = 1.2
VIEW_NOISE      = 0.1
CONVEX_RANDOM   = True
SEED            = 0
SKIP_MDT_DIRECT = True
MDT_DIRECT_MAX_TIME_BUDGET_SECONDS = 900
OUTPUT_CSV      = "tables/runtime_scaling/runtime_scaling.csv"


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
def make_two_view_blob_dataset(n_samples, centers, cluster_std, view_noise, random_state):
    X, y = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=cluster_std,
        n_features=2, random_state=random_state,
    )
    rng = np.random.default_rng(random_state)

    def rotation(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

    X_view_1 = X @ rotation(rng.uniform(0, np.pi)) + rng.normal(scale=view_noise, size=X.shape)
    X_view_2 = X @ rotation(rng.uniform(0, np.pi)) + rng.normal(scale=view_noise, size=X.shape)

    X_views      = [X_view_1, X_view_2]
    X_transition = [get_kernel_matrix(x, normalize=True)  for x in X_views]
    X_kernel     = [get_kernel_matrix(x, normalize=False) for x in X_views]
    return X_transition, X_kernel, X_views, y


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------
def _timed(fn):
    """Call fn(), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def _make_record(selection=0.0, operator=0.0, embedding=0.0, kmeans=0.0, selected_t=np.nan):
    pipeline = selection + operator + embedding + kmeans
    return {
        "selection_seconds":  selection,
        "operator_seconds":   operator,
        "embedding_seconds":  embedding,
        "kmeans_seconds":     kmeans,
        "pipeline_seconds":   pipeline,
        "total_seconds":      pipeline,
        "selected_t":         float(selected_t),
    }


# -----------------------------------------------------------------------------
# Run one repeat: operator → embed (k components) → KMeans
# -----------------------------------------------------------------------------
def run_one_repeat(build_fn, k, embed_method, run_seed):
    """
    Run one repeat: build operator → embed → KMeans.
    n_components is inferred from k (number of clusters).

    Parameters
    ----------
    build_fn    : callable(run_seed) -> (operator, timing_dict)
    k           : int — number of clusters, also used as n_components
    embed_method: str — one of "svd", "eigen", "truncated_svd", "precomputed"
    run_seed    : int
    """
    operator, timing = build_fn(run_seed)

    # n_components = k: consistent with spectral clustering convention
    emb, emb_sec = _timed(lambda: get_embedding(operator, k, method=embed_method))

    _, km_sec = _timed(
        lambda: KMeans(n_clusters=k, random_state=run_seed, n_init=10).fit(emb)
    )

    return _make_record(
        selection  = timing["selection_seconds"],
        operator   = timing["operator_seconds"],
        embedding  = emb_sec,
        kmeans     = km_sec,
        selected_t = timing["selected_t"],
    )


# -----------------------------------------------------------------------------
# Per-method build functions
# -----------------------------------------------------------------------------
def select_random_mdt_time(X, max_t):
    avg_operator = np.mean(np.asarray(X), axis=0)
    return int(get_diffusion_time(max_t=max_t, operator=avg_operator))


def _build_random_mdt(Xp, t, convex, run_seed):
    op, sec = _timed(lambda: random_mdt_operator(Xp, t=t, convex=convex))
    return op, _make_record(operator=sec)


def _build_random_mdt_with_selection(Xp, selection_max_t, convex):
    selected_t, sel_sec = _timed(lambda: select_random_mdt_time(Xp, max_t=selection_max_t))
    op, op_sec          = _timed(lambda: random_mdt_operator(Xp, t=selected_t, convex=convex))
    return op, _make_record(selection=sel_sec, operator=op_sec, selected_t=selected_t)


def _build_simple(fn, X):
    op, sec = _timed(lambda: fn(X))
    return op, _make_record(operator=sec)


def _build_contrastive(Xk, t):
    op, sec = _timed(lambda: mdt_contrastive(Xk, t=t))
    return op, _make_record(operator=sec)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def _summarize_runs(run_records):
    summary = {}

    def _safe_stats(values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        return (
            float(np.mean(finite)),
            float(np.std(finite)),
            float(np.min(finite)),
            float(np.max(finite)),
        )

    for key in run_records[0]:
        values = np.array([r[key] for r in run_records], dtype=float)
        mean_v, std_v, min_v, max_v = _safe_stats(values)
        summary[f"{key}_mean"] = mean_v
        summary[f"{key}_std"]  = std_v
        summary[f"{key}_min"]  = min_v
        summary[f"{key}_max"]  = max_v
    return summary


# -----------------------------------------------------------------------------
# Main benchmark loop
# -----------------------------------------------------------------------------
def benchmark_scaling(
    sizes, repeats, direct_repeats, t, selection_max_t,
    centers, cluster_std, view_noise, convex_random, seed,
    skip_mdt_direct=False, mdt_direct_time_budget_seconds=None,
    output_csv=None,
):
    """
    Benchmark runtime scaling of all methods across dataset sizes.

    Embedding method and n_components are determined per method:
      - Non-symmetric operators (random_mdt, mdt_contrastive): "svd"
      - Symmetric PSD operators (multiview, integrated, alternating): "eigen"
      - n_components = k (number of clusters) in all cases
    """
    rows = []
    mdt_direct_time_spent = 0.0

    for n_samples in sizes:
        print(f"\n{'='*52}\nN = {n_samples}\n{'='*52}")
        sys.stdout.flush()

        Xp, Xk, Xv, y = make_two_view_blob_dataset(
            n_samples=n_samples, centers=centers, cluster_std=cluster_std,
            view_noise=view_noise, random_state=seed + n_samples,
        )
        k = len(np.unique(y))

        # -----------------------------------------------------------------
        # Method registry: (name, build_fn, n_repeats, embed_method)
        #
        # embed_method is tied to operator type:
        #   "svd"   — stochastic / non-symmetric operators
        #   "eigen" — symmetric PSD operators
        #
        # n_components = k is passed at runtime inside run_one_repeat
        # -----------------------------------------------------------------
        methods = [
            (
                "random_mdt",
                lambda rs: _build_random_mdt(Xp, t=t, convex=convex_random, run_seed=rs),
                repeats,
                "svd",
            ),
            (
                "random_mdt_with_selection",
                lambda rs: _build_random_mdt_with_selection(Xp, selection_max_t, convex_random),
                repeats,
                "svd",
            ),
            # (
            #     "mdt_contrastive",
            #     lambda rs: _build_contrastive(Xp, t=t),
            #     repeats,
            #     "svd",
            # ),
            (
                "multiview_diffusion",
                lambda rs: _build_simple(multiview_diffusion, Xk),
                repeats,
                "eigen",
            ),
            (
                "integrated_diffusion",
                lambda rs: _build_simple(integrated_diffusion, Xp),
                repeats,
                "svd",
            ),
            (
                "alternating_diffusion",
                lambda rs: _build_simple(alternating_diffusion, Xp),
                repeats,
                "svd",
            ),
        ]

        if not skip_mdt_direct:
            methods.insert(3, (
                "mdt_direct",
                lambda rs: _build_simple(
                    lambda Xp: mdt_direct(Xp, t=t, k=k, metric="chs"), Xp
                ),
                direct_repeats,
                "svd",
            ))

        # -----------------------------------------------------------------
        # Run each method
        # -----------------------------------------------------------------
        for method_name, build_fn, n_rep, embed_method in methods:

            # Budget guard for mdt_direct
            over_budget = (
                method_name == "mdt_direct"
                and mdt_direct_time_budget_seconds is not None
                and mdt_direct_time_spent >= mdt_direct_time_budget_seconds
            )
            if over_budget:
                print(
                    f"Skipping {method_name}: budget exhausted "
                    f"({mdt_direct_time_spent:.1f}s / {mdt_direct_time_budget_seconds:.1f}s)"
                )
                sys.stdout.flush()
                continue

            run_records = []

            for run_idx in range(n_rep):
                run_seed = seed + run_idx
                np.random.seed(run_seed)

                try:
                    record = run_one_repeat(
                        build_fn,
                        k=k,
                        embed_method=embed_method,
                        run_seed=run_seed,
                    )
                except Exception as e:
                    print(f"  [{method_name}] run {run_idx} FAILED: {e}")
                    sys.stdout.flush()
                    continue

                run_records.append(record)
                print(
                    f"  [{method_name}] run {run_idx}: "
                    f"operator={record['operator_seconds']:.3f}s  "
                    f"embed={record['embedding_seconds']:.3f}s  "
                    f"kmeans={record['kmeans_seconds']:.3f}s  "
                    f"total={record['total_seconds']:.3f}s"
                )
                sys.stdout.flush()

                if method_name == "mdt_direct":
                    mdt_direct_time_spent += record["pipeline_seconds"]
                    if (
                        mdt_direct_time_budget_seconds is not None
                        and mdt_direct_time_spent >= mdt_direct_time_budget_seconds
                    ):
                        print("  Budget reached mid-run, stopping mdt_direct.")
                        sys.stdout.flush()
                        break

            if not run_records:
                continue

            row = {
                "n_samples":    n_samples,
                "method":       method_name,
                "embed_method": embed_method,
                "n_components": k,
                "n_runs":       len(run_records),
            }
            row.update(_summarize_runs(run_records))
            rows.append(row)

            # Save after every method — crash-safe
            if output_csv is not None:
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                pd.DataFrame(rows).to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")
    return df


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    results = benchmark_scaling(
        sizes=SIZES,
        repeats=REPEATS,
        direct_repeats=DIRECT_REPEATS,
        t=TRAJECTORY_T,
        selection_max_t=SELECTION_MAX_T,
        centers=CENTERS,
        cluster_std=CLUSTER_STD,
        view_noise=VIEW_NOISE,
        convex_random=CONVEX_RANDOM,
        seed=SEED,
        skip_mdt_direct=SKIP_MDT_DIRECT,
        mdt_direct_time_budget_seconds=MDT_DIRECT_MAX_TIME_BUDGET_SECONDS,
        output_csv=OUTPUT_CSV,
    )
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
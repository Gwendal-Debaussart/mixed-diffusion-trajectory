import argparse
import os
import sys
import time
from typing import cast

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from benchmarks.load_dataset import get_num_clusters, load_preprocessed_dataset
from experiment_utils.get_diffusion_time import get_diffusion_time
from experiment_utils.get_operator_from_method import get_operator_from_method
from experiment_utils.method_list import method_list
from experiment_utils.method_to_embedding import method_to_embedding

from competitors.alternating_diffusion import alternating_diffusion, powered_alternating_diffusion
from competitors.composite_diffusion import composite_diffusion_operator
from competitors.cross_diffusion import cross_diffusion_operator
from competitors.gcca import gcca_embedding
from competitors.integrated_diffusion import integrated_diffusion
from competitors.multiview_diffusion import multiview_diffusion
from mdt.random_mdt import random_mdt_operator
from mdt.mdt_direct import mdt_direct
from mdt.mdt_contrastive import mdt_contrastive
from mdt.mdt_tree import mdt_beam

DEFAULT_DATASETS = [
    {"name": "isolet_lindenbaum"},
    {"name": "multiple_feat"},
    {"name": "olivetti"},
    {"name": "caltech101-7"},
    {"name": "leaves"},
    {"name": "yale"},
    {"name": "msrc"},
]

DEFAULT_RAW_DIR = "tables/runtime_benchmark_raw"
DEFAULT_SUMMARY_DIR = "tables/runtime_benchmark"

TIMING_KEYS = [
    "operator_seconds",
    "embedding_seconds",
    "kmeans_seconds",
    "pipeline_seconds",
    "total_seconds",
]

def _method_list():
    return (
        {
            "name": "Alternating Diffusion",
            "func": alternating_diffusion,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "Multi-view Diffusion Maps",
            "func": multiview_diffusion,
            "input_type": "kernels",
            "decomp_method": "eigen",
        },
        {
            "name": "Single-view Diffusion Maps",
            "func": lambda X: X[0],
            "input_type": "preprocessed",
            "single_view": True,
            "decomp_method": "eigen",
        },
        {
            "name": "Integrated Diffusion Maps",
            "func": integrated_diffusion,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "Composite Diffusion Maps",
            "func": composite_diffusion_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "n_views": 2,
        },
        {
            "name": "Cross Diffusion Maps",
            "func": cross_diffusion_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "GCCA",
            "func": gcca_embedding,
            "params": lambda dn: {"n_components": get_num_clusters(dn)},
            "input_type": "preprocessed",
            "decomp_method": "precomputed",
        },
        {
            "name": "Powered Alternating Diffusion",
            "func": powered_alternating_diffusion,
            "input_type": "preprocessed",
            "decomp_method": "svd",
        },
        {
            "name": "Random Convex MDT",
            "func": random_mdt_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "stochastic": True,
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
                "convex": True,
            },
        },
        {
            "name": "Random MDT",
            "func": random_mdt_operator,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "stochastic": True,
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
                "convex": False,
            },
        },
        {
            "name": "Direct MDT",
            "func": mdt_direct,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
                "k": get_num_clusters(dn),
            },
            "task" : "clustering"
        },
        {
            "name": "Contrastive MDT",
            "func": mdt_contrastive,
            "input_type": "preprocessed",
            "task": "manifold_learning",
            "decomp_method": "svd",
            "params": lambda dn: {
                "t": get_diffusion_time(dn),
            },
        },
        {
            "name": "Beam-Search MDT",
            "func": mdt_beam,
            "input_type": "preprocessed",
            "decomp_method": "svd",
            "params": lambda dn: {
                "n_cluster": get_num_clusters(dn),
                "max_depth": 2*get_diffusion_time(dn),
            },
            "task" : "clustering"
        },
    )

def _dataset_key(dataset):
    key = dataset["name"]
    if "noise_factor" in dataset and dataset["noise_factor"] is not None:
        key += f"_noise_{dataset['noise_factor']:.2f}"
    return key


def _timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _make_record(operator=0.0, embedding=0.0, kmeans=0.0):
    pipeline = operator + embedding + kmeans
    return {
        "operator_seconds": float(operator),
        "embedding_seconds": float(embedding),
        "kmeans_seconds": float(kmeans),
        "pipeline_seconds": float(pipeline),
        "total_seconds": float(pipeline),
    }


def _record_to_row(dataset, method_name, repeat, record):
    return {
        "dataset": dataset["name"],
        "noise_factor": dataset.get("noise_factor", np.nan),
        "method": method_name,
        "repeat": repeat,
        "operator_seconds": record["operator_seconds"],
        "embedding_seconds": record["embedding_seconds"],
        "kmeans_seconds": record["kmeans_seconds"],
        "pipeline_seconds": record["pipeline_seconds"],
        "total_seconds": record["total_seconds"],
    }


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

    for key in TIMING_KEYS:
        values = np.array([record[key] for record in run_records], dtype=float)
        mean_v, std_v, min_v, max_v = _safe_stats(values)
        summary[f"{key}_mean"] = mean_v
        summary[f"{key}_std"] = std_v
        summary[f"{key}_min"] = min_v
        summary[f"{key}_max"] = max_v
    return summary


def _summarize_raw_file(filepath):
    df = pd.read_csv(filepath)
    required_cols = {"dataset", "method", "repeat", *TIMING_KEYS}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw runtime file {filepath}: {sorted(missing)}")

    # Keep only rows with valid method names and numeric timing values.
    df = df.dropna(subset=["dataset", "method", "repeat"])
    for col in TIMING_KEYS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=TIMING_KEYS)

    if df.empty:
        return pd.DataFrame()

    summary_rows = []
    grouped = df.groupby(["dataset", "method"], dropna=False)
    for (dataset_name, method_name), grp in grouped:
        run_records = [
            {
                "operator_seconds": float(row["operator_seconds"]),
                "embedding_seconds": float(row["embedding_seconds"]),
                "kmeans_seconds": float(row["kmeans_seconds"]),
                "pipeline_seconds": float(row["pipeline_seconds"]),
                "total_seconds": float(row["total_seconds"]),
            }
            for _, row in grp.iterrows()
        ]
        summary = {
            "dataset": dataset_name,
            "noise_factor": grp["noise_factor"].iloc[0] if "noise_factor" in grp.columns else np.nan,
            "method": method_name,
            "n_runs": int(pd.to_numeric(grp["repeat"], errors="coerce").nunique()),
        }
        summary.update(_summarize_runs(run_records))
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)


def _build_summary_from_raw(datasets, save_dir=DEFAULT_RAW_DIR):
    summary_parts = []
    for dataset in datasets:
        raw_path = os.path.join(save_dir, f"{_dataset_key(dataset)}.csv")
        if not os.path.exists(raw_path):
            continue
        summary_df = _summarize_raw_file(raw_path)
        if not summary_df.empty:
            summary_parts.append(summary_df)

    if not summary_parts:
        return pd.DataFrame()

    result = pd.concat(summary_parts, ignore_index=True)
    result = result.sort_values(["dataset", "method"]).reset_index(drop=True)
    return result


def _get_existing_repeats(dataset, method_name, save_dir=DEFAULT_RAW_DIR):
    filepath = os.path.join(save_dir, f"{_dataset_key(dataset)}.csv")
    if not os.path.exists(filepath):
        return 0

    df = pd.read_csv(filepath)
    if df.empty:
        return 0

    existing = df[df["method"] == method_name]
    if existing.empty:
        return 0

    repeats = [float(value) for value in existing["repeat"].tolist() if pd.notna(value)]
    return int(max(repeats)) + 1 if repeats else 0


def _save_raw_results(dataset, method_name, run_records, save_dir=DEFAULT_RAW_DIR):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{_dataset_key(dataset)}.csv")

    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        if df_existing.empty:
            repeat_offset = 0
        else:
            existing = df_existing[df_existing["method"] == method_name]
            if existing.empty:
                repeat_offset = 0
            else:
                repeats = [float(value) for value in existing["repeat"].tolist() if pd.notna(value)]
                repeat_offset = int(max(repeats)) + 1 if repeats else 0
    else:
        repeat_offset = 0

    rows = [
        _record_to_row(dataset, method_name, idx + repeat_offset, record)
        for idx, record in enumerate(run_records)
    ]
    df_new = pd.DataFrame(rows)

    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df_new.to_csv(filepath, index=False)

    print(f"[{dataset['name']}] Appended runtime rows to {filepath}")


def _load_dataset(dataset):
    dataset_name = dataset["name"]
    noise_factor = dataset.get("noise_factor", None)
    if noise_factor is not None:
        loaded = load_preprocessed_dataset(
            dataset_name, return_views=True, noise_factor=noise_factor
        )
    else:
        loaded = load_preprocessed_dataset(dataset_name, return_views=True)
    return cast(tuple[list[np.ndarray], list[np.ndarray], np.ndarray], loaded)


def _precompute_metadata(datasets, n_jobs=-1):
    def compute_for_dataset(dataset):
        kwargs = {}
        if "noise_factor" in dataset:
            kwargs["noise_factor"] = dataset["noise_factor"]
        diffusion_time = get_diffusion_time(dataset["name"], **kwargs)
        num_clusters = get_num_clusters(dataset["name"])
        print(f"[{dataset['name']}] diffusion time={diffusion_time}, clusters={num_clusters}")

    Parallel(n_jobs=n_jobs)(delayed(compute_for_dataset)(dataset) for dataset in datasets)


def _run_one_repeat(method, dataset_name, num_clusters, dim_embedd, X_preprocessed, X_views, run_seed):
    try:
        np.random.seed(run_seed)
        operator, operator_seconds = _timed(
            lambda: get_operator_from_method(method, dataset_name, X_preprocessed, X_views)
        )
        embedding, embedding_seconds = _timed(
            lambda: method_to_embedding(operator, X_views, method, dim_embedd)
        )

        n_samples = len(X_views[0]) if X_views else len(X_preprocessed[0])
        if isinstance(embedding, np.ndarray) and embedding.shape[0] > n_samples:
            embedding = embedding[:n_samples, :]

        _, kmeans_seconds = _timed(
            lambda: KMeans(n_clusters=num_clusters, random_state=run_seed, n_init=10).fit(embedding)
        )

        return {
            "ok": True,
            "record": _make_record(
                operator=operator_seconds,
                embedding=embedding_seconds,
                kmeans=kmeans_seconds,
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _split_run_outcomes(run_outcomes):
    run_records = []
    failed_errors = []
    for outcome in run_outcomes:
        if outcome.get("ok"):
            record = outcome.get("record")
            if isinstance(record, dict):
                run_records.append(record)
        else:
            failed_errors.append(str(outcome.get("error", "Unknown error")))
    return run_records, failed_errors


def benchmark_runtimes_for_dataset(
    dataset,
    methods,
    num_repeats=3,
    n_jobs=-1,
    save_dir=DEFAULT_RAW_DIR,
):
    dataset_name = dataset["name"]
    noise_factor = dataset.get("noise_factor", None)
    print(f"\n[{dataset_name}] Loading dataset...")
    X_preprocessed, X_views, Y = _load_dataset(dataset)
    num_clusters = int(len(np.unique(Y)))
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
                method_copy["name"] = f"{method['name']} (view {view_idx + 1})"
                existing_repeats = _get_existing_repeats(dataset, method_copy["name"], save_dir)
                repeats_needed = num_repeats - existing_repeats
                if repeats_needed <= 0:
                    print(
                        f"[{dataset_name}] (nf = {noise_factor}) already computed [{method_copy['name']}]"
                    )
                    continue

                Xp_view = [X_preprocessed[view_idx]]
                Xv_view = [X_views[view_idx]]
                run_outcomes = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(_run_one_repeat)(
                        method_copy,
                        dataset_name,
                        num_clusters,
                        dim_embedd,
                        Xp_view,
                        Xv_view,
                        run_seed=i,
                    )
                    for i in range(existing_repeats, existing_repeats + repeats_needed)
                )
                run_records, failed_errors = _split_run_outcomes(run_outcomes)
                if failed_errors:
                    print(
                        f"[{dataset_name}] {method_copy['name']}: {len(failed_errors)} run(s) failed; first error: {failed_errors[0]}"
                    )
                if not run_records:
                    print(f"[{dataset_name}] {method_copy['name']}: no successful runs, skipping save")
                    continue
                _save_raw_results(dataset, method_copy["name"], run_records, save_dir)
                for idx, record in enumerate(run_records):
                    all_results.append(
                        _record_to_row(
                            dataset,
                            method_copy["name"],
                            idx + existing_repeats,
                            record,
                        )
                    )
                print(
                    f"[{dataset_name}] Completed [{repeats_needed}] repeats for {method_copy['name']}"
                )
            continue

        existing_repeats = _get_existing_repeats(dataset, method["name"], save_dir)
        repeats_needed = num_repeats - existing_repeats
        if repeats_needed <= 0:
            print(f"[{dataset_name}] (nf = {noise_factor}) {method['name']}: already has {num_repeats} runs")
            continue

        print(
            f"[{dataset_name}] (n.f = {noise_factor}) Running {method['name']} ({repeats_needed} missing repeats)"
        )
        run_outcomes = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_one_repeat)(
                method,
                dataset_name,
                num_clusters,
                dim_embedd,
                X_preprocessed,
                X_views,
                run_seed=i,
            )
            for i in range(existing_repeats, existing_repeats + repeats_needed)
        )
        run_records, failed_errors = _split_run_outcomes(run_outcomes)
        if failed_errors:
            print(
                f"[{dataset_name}] {method['name']}: {len(failed_errors)} run(s) failed; first error: {failed_errors[0]}"
            )
        if not run_records:
            print(f"[{dataset_name}] {method['name']}: no successful runs, skipping save")
            continue
        _save_raw_results(dataset, method["name"], run_records, save_dir)
        for idx, record in enumerate(run_records):
            all_results.append(
                _record_to_row(dataset, method["name"], idx + existing_repeats, record)
            )
        print(f"[{dataset_name}] Completed [{repeats_needed}] repeats for {method['name']}")

    if not all_results:
        return pd.DataFrame()

    summary_rows = []
    grouped = {}
    for row in all_results:
        grouped.setdefault((row["dataset"], row["method"]), []).append(row)

    for (dataset_name, method_name), rows in grouped.items():
        summary = {
            "dataset": dataset_name,
            "noise_factor": rows[0].get("noise_factor", np.nan),
            "method": method_name,
            "n_runs": len(rows),
        }
        summary.update(_summarize_runs(rows))
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)


def benchmark_runtimes(
    datasets,
    methods=None,
    num_repeats=3,
    n_jobs=-1,
    save_dir=DEFAULT_RAW_DIR,
    summary_dir=DEFAULT_SUMMARY_DIR,
):
    if methods is None:
        methods = method_list()

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    _precompute_metadata(datasets, n_jobs=n_jobs)

    all_summary = []
    for dataset in datasets:
        summary_df = benchmark_runtimes_for_dataset(
            dataset,
            methods,
            num_repeats=num_repeats,
            n_jobs=n_jobs,
            save_dir=save_dir,
        )
        if not summary_df.empty:
            all_summary.append(summary_df)

    # Always rebuild summary from raw files so reruns include previously computed methods.
    result = _build_summary_from_raw(datasets=datasets, save_dir=save_dir)
    summary_path = os.path.join(summary_dir, "runtime_dataset.csv")
    result.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    return result


def _build_default_datasets(include_mnist_noises=True):
    datasets = [
        {"name": "isolet_lindenbaum"},
        {"name": "multiple_feat"},
        {"name": "olivetti"},
        {"name": "caltech101-7"},
        {"name": "leaves"},
        {"name": "yale"},
        {"name": "msrc"},
    ]
    if include_mnist_noises:
        return datasets
    return [dataset for dataset in datasets if "noise_factor" not in dataset]


def _build_parser():
    parser = argparse.ArgumentParser(description="Benchmark method runtimes across datasets.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per method and dataset.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for dataset metadata and repeat execution.",
    )
    parser.add_argument(
        "--save-dir",
        default=DEFAULT_RAW_DIR,
        help=f"Directory for raw runtime CSVs (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--summary-dir",
        default=DEFAULT_SUMMARY_DIR,
        help=f"Directory for summary CSVs (default: {DEFAULT_SUMMARY_DIR})",
    )
    parser.add_argument(
        "--no-mnist-noise",
        action="store_true",
        help="Exclude the noisy MNIST datasets from the default dataset list.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    datasets = _build_default_datasets(include_mnist_noises=not args.no_mnist_noise)
    results = benchmark_runtimes(
        datasets=datasets,
        methods=_method_list(),
        num_repeats=args.repeats,
        n_jobs=args.n_jobs,
        save_dir=args.save_dir,
        summary_dir=args.summary_dir,
    )
    if not results.empty:
        print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

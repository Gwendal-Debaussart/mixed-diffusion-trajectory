import os
import csv
import numpy as np
import pandas as pd
from benchmarks.load_dataset import load_preprocessed_dataset
from utilities.entropy import entropy_from_values


def _sanitize_cache_key(key: str) -> str:
    return str(key).replace(os.sep, "_").replace(" ", "_")


def _default_cache_key(dataset_name=None, noise_factor=None, cache_key=None):
    if cache_key is not None:
        return _sanitize_cache_key(cache_key)
    if dataset_name is None:
        return None

    suffix = ""
    if noise_factor is not None:
        suffix = f"_noise{noise_factor:.2f}"
    elif dataset_name in ["mnist_kuchroo", "mnist_lindenbaum"]:
        suffix = "_noise0.50"
    return f"{dataset_name}{suffix}"


def compute_entropy(dataset_name=None, t_max=50, operator=None, cache_key=None, **kwargs):
    """
    Run singular entropy experiments for a dataset, resuming from existing results.

    - Never overwrites existing data.
    - One file per dataset and noise factor.
    - Computes only missing t values (default 1..25).
    - Saves after every step (safe for interruption).

    Args:
        dataset_name (str | None): Name of the dataset.
        t_max (int): Maximum diffusion time to evaluate.
        operator (np.ndarray | None): Optional precomputed operator. If provided,
            dataset loading is skipped.
        cache_key (str | None): Optional key for csv caching. If None and
            dataset_name is None, no file is written/read.
        **kwargs: Extra args for `load_preprocessed_dataset`.
                  May include 'noise_factor' (float).
    """
    if dataset_name is None and operator is None:
        raise ValueError("Must provide either dataset_name or operator.")

    save_dir = "tables/singular_entropy"
    os.makedirs(save_dir, exist_ok=True)

    nf = kwargs.get("noise_factor")
    if nf is not None:
        if not isinstance(nf, (int, float)):
            raise ValueError("noise_factor must be numeric if provided.")

    key = _default_cache_key(dataset_name=dataset_name, noise_factor=nf, cache_key=cache_key)
    filepath = os.path.join(save_dir, f"{key}.csv") if key is not None else None

    if operator is None:
        loaded = load_preprocessed_dataset(dataset_name, **kwargs)
        X = loaded[0]
        operator = np.mean(X, axis=0)
    else:
        operator = np.asarray(operator)
        if operator.ndim != 2:
            raise ValueError("operator must be a 2D square matrix.")

    last_t = 0
    file_exists = filepath is not None and os.path.isfile(filepath)
    if file_exists and filepath is not None:
        try:
            df = pd.read_csv(filepath)
            if not df.empty:
                last_t = int(df["t"].max())
        except Exception as e:
            label = key if key is not None else (dataset_name or "operator")
            print(f"[{label}] Warning: could not read existing file ({e}).")

    if last_t >= t_max:
        entropies = df[df["t"] <= t_max].copy()
        return entropies

    running_operator = np.linalg.matrix_power(operator, last_t+1)
    new_rows = []
    for t in range(last_t+1, t_max + 1):
        singular_vals = np.linalg.svd(running_operator, compute_uv=False)
        singular_vals = np.abs(singular_vals)
        new_rows.append([t, entropy_from_values(singular_vals)])
        running_operator = running_operator @ operator

    if filepath is None:
        entropies = pd.DataFrame(new_rows, columns=["t", "singular_entropy"])
    else:
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["t", "singular_entropy"])
            writer.writerows(new_rows)
        entropies = pd.read_csv(filepath)

    label = key if key is not None else (dataset_name or "operator")
    print(f"[{label}] Singular entropy computed up to t={t_max}.")
    return entropies
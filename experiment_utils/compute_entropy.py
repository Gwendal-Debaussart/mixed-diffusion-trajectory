import os
import csv
import numpy as np
import pandas as pd
from benchmarks.load_dataset import load_preprocessed_dataset
from utilities.entropy import powered_singular_entropy


def compute_entropy(dataset_name: str, t_max = 50, **kwargs):
    """
    Run singular entropy experiments for a dataset, resuming from existing results.

    - Never overwrites existing data.
    - One file per dataset and noise factor.
    - Computes only missing t values (default 1..25).
    - Saves after every step (safe for interruption).

    Args:
        dataset_name (str): Name of the dataset.
        **kwargs: Extra args for `load_preprocessed_dataset`.
                  May include 'noise_factor' (float) to distinguish runs.
    """
    save_dir = "tables/singular_entropy"
    os.makedirs(save_dir, exist_ok=True)

    noise_suffix = ""
    if "noise_factor" in kwargs:
        nf = kwargs["noise_factor"]
        if not isinstance(nf, (int, float)):
            raise ValueError("noise_factor must be numeric if provided.")
        noise_suffix = f"_noise{nf:.2f}"
    elif dataset_name in ["mnist_kuchroo", "mnist_lindenbaum"]:
        nf = 0.5
        noise_suffix = f"_noise{nf:.2f}"

    filepath = os.path.join(save_dir, f"{dataset_name}{noise_suffix}.csv")

    last_t = 0
    file_exists = os.path.isfile(filepath)
    if file_exists:
        try:
            df = pd.read_csv(filepath)
            if not df.empty:
                last_t = int(df["t"].max())
        except Exception as e:
            print(
                f"[{dataset_name}] Warning: could not read existing file ({e})."
            )

    loaded = load_preprocessed_dataset(dataset_name, **kwargs)
    X = loaded[0]
    operator = np.mean(X, axis=0)
    singular_vals = np.linalg.svd(operator, compute_uv=False)
    singular_vals = np.abs(singular_vals)

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["t", "singular_entropy"])

        for t in range(last_t + 1, t_max + 1):
            val = powered_singular_entropy(singular_vals, t)
            writer.writerow([t, val])
    entropies = pd.read_csv(filepath)
    print(f"[{dataset_name}] Singular entropy computed up to t={t_max}.")
    return entropies
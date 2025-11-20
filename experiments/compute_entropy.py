import os
import csv
import numpy as np
import pandas as pd
from utilities.entropy import singular_entropy
from benchmarks.load_dataset import load_preprocessed_dataset


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
                f"[{dataset_name}] Warning: could not read existing file ({e}), starting fresh."
            )

    X, _ = load_preprocessed_dataset(dataset_name, **kwargs)
    operator = np.mean(X, axis=0)
    running_operator = operator.copy()
    if last_t > 0:
        running_operator = np.linalg.matrix_power(operator, last_t+1)

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["t", "singular_entropy"])

        for t in range(last_t + 1, t_max + 1):
            print(f"[{dataset_name}] Resuming from t={last_t} ({filepath})")
            val = singular_entropy(running_operator)
            writer.writerow([t, val])
            running_operator = running_operator @ operator
            print(f"[{dataset_name}] t={t} done, saved to {os.path.basename(filepath)}")
    entropies = pd.read_csv(filepath)
    return entropies
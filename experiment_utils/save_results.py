import os
import pandas as pd
import csv
import numpy as np


def save_raw_results(
    dataset, method_name, repeats, save_dir="tables/clustering_benchmark_raw/"
):
    """
    Save raw repeat-level benchmark results (append-only).
    """
    os.makedirs(save_dir, exist_ok=True)
    fname = dataset["name"]
    if "noise_factor" in dataset:
        fname += f"_noise_{dataset['noise_factor']:.2f}"
    filepath = os.path.join(save_dir, f"{fname}.csv")

    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        mask = df_existing["method"] == method_name
        if mask.any():
            repeat_offset = df_existing.loc[mask, "repeat"].max() + 1
        else:
            repeat_offset = 0
    else:
        repeat_offset = 0

    rows = []
    for i, repeat in enumerate(repeats):
        for metric, value in repeat.items():
            rows.append(
                {
                    "method": method_name,
                    "repeat": i + repeat_offset,
                    "metric": metric,
                    "value": value,
                }
            )
    df_new = pd.DataFrame(rows)

    if os.path.exists(filepath):
        df_new.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df_new.to_csv(filepath, index=False)

    print(f"Raw results appended to {filepath}")


def format_results(
    dataset_name, noise_factor=None, input_dir="tables/clustering_benchmark_raw/"
):
    """
    Format raw benchmark results into summary statistics.

    Args:
        dataset_name (str): Name of the dataset.
        noise_factor (float, optional): Noise factor if applicable.
        save_dir (str): Directory containing raw CSVs.

    Returns:
        pd.DataFrame: Summary dataframe with columns:
                      ['method', 'metric', 'mean', 'std', 'n_repeats']
    """
    fname = dataset_name
    if noise_factor is not None:
        fname += f"_noise_{noise_factor:.2f}"
    filepath = os.path.join(input_dir, f"{fname}.csv")

    if not os.path.exists(filepath):
        print(f"[{dataset_name}] Raw results file not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    summary_rows = []
    grouped = df.groupby(["method", "metric"])
    for (method, metric), group in grouped:
        values = group["value"].values
        summary_rows.append(
            {
                "method": method,
                "metric": metric,
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "n_repeats": len(values),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_dir = os.path.join(os.path.dirname(input_dir), "../benchmark_results")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"{fname}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[{dataset_name}] Summary saved to {summary_path}")
    return summary_df

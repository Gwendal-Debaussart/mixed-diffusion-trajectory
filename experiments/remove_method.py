import os
import pandas as pd

def remove_method_from_dataset(
    method_name,
    dataset_name,
    noise_factor=None,
    save_dir="tables/clustering_benchmark_raw/",
):
    """
    Remove all saved results for a given method and dataset.
    Parameters:
    - method_name: str, name of the method
    - dataset_name: str, name of the dataset
    - save_dir: str, directory where results are saved
    """

    if noise_factor is not None:
        dataset_name = f"{dataset_name}_noise_{noise_factor}"

    file_path = os.path.join(save_dir, f"{dataset_name}.csv")

    if not os.path.exists(file_path):
        print(f"No CSV found for {dataset_name} at {save_dir}.")
        return

    df = pd.read_csv(file_path)

    df_filtered = df[df['method'] != method_name]

    if len(df_filtered) < len(df):
        df_filtered.to_csv(file_path, index=False)
        print(f"Removed method '{method_name}' from {dataset_name}.")
    else:
        print(f"No entries found for method '{method_name}' in {dataset_name}.")

def remove_method_from_all_datasets(
    method_name,
    datasets,
    save_dir="tables/clustering_benchmark_raw/",
):
    """
    Remove all saved results for a given method across multiple datasets.
    Parameters:
    - method_name: str, name of the method
    - datasets: list of dict, each dict contains "name" of the dataset and optional "noise_factor"
    - save_dir: str, directory where results are saved
    """
    for dataset in datasets:
        remove_method_from_dataset(
            method_name,
            dataset["name"],
            noise_factor=dataset.get("noise_factor", None),
            save_dir=save_dir,
        )
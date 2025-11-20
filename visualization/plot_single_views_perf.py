import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from .style import get_col_list

def plot_single_views_perf(dataset_name, show = False):
    """
    Plots the performance of single views for a given dataset.
    Args:
        dataset_name (str): Name of the dataset.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
    os.makedirs("figures/single_views_perf", exist_ok=True)

    if os.path.isfile(f"tables/benchmark_results/{dataset_name}.csv") is False:
        raise FileNotFoundError(f"No benchmark results for {dataset_name} found.")

    perf_df = pd.read_csv(f"tables/benchmark_results/{dataset_name}.csv")
    perf_df = perf_df[perf_df["metric"] == "ami"].copy()
    mask = perf_df['method'].str.startswith('Single-view Diffusion Maps')
    single_views = perf_df.loc[mask].copy()
    single_views['view'] = single_views['method'].str.extract(r'\(view\s*(\d+)\)').astype(int)
    single_views = single_views.sort_values("view")

    plt.bar(single_views["view"], single_views["mean"], color= get_col_list(), capsize=4)
    plt.xlabel("View")
    plt.xticks(single_views["view"])
    plt.ylabel("AMI")
    plt.tight_layout()
    plt.savefig(
        f"figures/single_views_perf/{dataset_name}.pdf",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()
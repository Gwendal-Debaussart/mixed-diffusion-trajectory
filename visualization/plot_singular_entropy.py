import pandas as pd
import matplotlib.pyplot as plt
import os
from experiments.compute_entropy import compute_entropy
from experiments.get_diffusion_time import get_diffusion_time
from .style import *


def plot_singular_entropy(
    dataset_name,
    noise_factor=None,
    max_time=None,
    elbow = False,
    path="tables/singular_entropy",
    save=False,
):
    """
    Plots the singular entropy stored in the corresponding CSV file for the given dataset.
    Args:
        dataset_name (str): Name of the dataset.
        noise_factor (float, optional): Noise factor used in the dataset. Defaults to None.
        save (bool, optional): Whether to save the plot as a PDF. Defaults to False.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"
    os.makedirs("figures/singular_entropy", exist_ok=True)
    noise_suffix = ""
    if dataset_name in ["mnist_lindenbaum", "mnist_kuchroo"]:
        if noise_factor is None:
            nf = 0.5
        else:
            nf = noise_factor
        noise_suffix = f"_noise{nf:.2f}"
    if os.path.isfile(f"{path}/{dataset_name}{noise_suffix}.csv") is False:
        if noise_factor:
            compute_entropy(dataset_name, noise_factor=noise_factor)
        else:
            compute_entropy(dataset_name)
    entropies = pd.read_csv(f"{path}/{dataset_name}{noise_suffix}.csv")
    entropies = entropies[entropies["t"] <= max_time] if max_time else entropies
    if elbow:
        elbow_time = get_diffusion_time(dataset_name, max_t=max_time, noise_factor=noise_factor)
        plt.axvline(x=elbow_time, color="#888888", ls="--", label=f"Elbow at t={elbow_time}")
    plt.plot(
        entropies["t"],
        entropies["singular_entropy"],
        ls="-",
        marker="o",
        color=get_col_list()[0],
        label=r"Singular Entropy of $\mathbb{E}[\mathrm{W}^{(t)}]$",
    )
    plt.xlabel(r"Time step $t$")
    plt.ylabel("Singular Entropy")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(
            f"figures/singular_entropy/{dataset_name}{noise_suffix}_singular_entropy.pdf",
            bbox_inches="tight",
        )
    plt.show()


import os
import pandas as pd
import numpy as np
from utilities.entropy import singular_entropy
from kneed import KneeLocator
from benchmarks.load_dataset import load_preprocessed_dataset
from .compute_entropy import compute_entropy

def get_diffusion_time(dataset_name, max_t = 50, **kwargs):
    """
    Returns the diffusion time parameter for a given dataset.
    """
    os.makedirs("tables/singular_entropy", exist_ok=True)
    nf = kwargs.get("noise_factor")
    if nf is not None:
        key = f"{dataset_name}_noise_{nf:.2f}"
    else:
        key = dataset_name

    if os.path.exists("tables/singular_entropy/diffusion_times.csv"):
        df = pd.read_csv("tables/singular_entropy/diffusion_times.csv", index_col=0)
        if key in df.index:
            return int(df.loc[key, "diffusion_time"])
    else:
        print("Diffusion times file not found.")
        df = pd.DataFrame(columns=["dataset", "diffusion_time"])
        df.to_csv("tables/singular_entropy/diffusion_times.csv")

    # Default diffusion time if not specified
    entropies_df = compute_entropy(dataset_name, **kwargs)
    y = entropies_df["singular_entropy"].values
    y = y[:max_t] #ensures that the knee is not considered too far, 20/25 seems good enough, but is highly dependant on datasets !!
    knee_locator = KneeLocator(range(1, len(y)+1), y, curve="convex", direction="decreasing")
    if knee_locator.knee is None:
        t = 1
    else:
        t = knee_locator.knee
    df = pd.read_csv("tables/singular_entropy/diffusion_times.csv", index_col=0)
    df.loc[key] = t
    df.to_csv("tables/singular_entropy/diffusion_times.csv")

    return t

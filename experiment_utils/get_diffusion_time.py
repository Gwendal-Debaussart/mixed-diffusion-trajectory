import os
import pandas as pd
from kneed import KneeLocator
from .compute_entropy import compute_entropy


def _diffusion_key(dataset_name=None, noise_factor=None, cache_key=None):
    if cache_key is not None:
        return str(cache_key)
    if dataset_name is None:
        return None
    if noise_factor is not None:
        return f"{dataset_name}_noise_{noise_factor:.2f}"
    return dataset_name


def get_diffusion_time(dataset_name=None, max_t=50, operator=None, cache_key=None, **kwargs):
    """
    Returns the diffusion time parameter for a dataset or a provided operator.
    """
    if dataset_name is None and operator is None:
        raise ValueError("Must provide either dataset_name or operator.")

    os.makedirs("tables/singular_entropy", exist_ok=True)
    nf = kwargs.get("noise_factor")
    key = _diffusion_key(dataset_name=dataset_name, noise_factor=nf, cache_key=cache_key)

    df = None
    csv_path = "tables/singular_entropy/diffusion_times.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        if key is not None and key in df.index:
            return int(pd.to_numeric(df.loc[key, "diffusion_time"], errors="coerce"))
    else:
        df = pd.DataFrame(columns=["dataset", "diffusion_time"])
        df.to_csv(csv_path)

    entropies_df = compute_entropy(
        dataset_name=dataset_name,
        t_max=max_t,
        operator=operator,
        cache_key=key,
        **kwargs,
    )
    y = entropies_df["singular_entropy"].values
    y = y[:max_t]
    knee_locator = KneeLocator(range(1, len(y)+1), y, curve="convex", direction="decreasing")
    if knee_locator.knee is None:
        t = 1
    else:
        t = knee_locator.knee

    if key is not None:
        if df is None:
            df = pd.read_csv(csv_path, index_col=0)
        df.loc[key] = t
        df.to_csv(csv_path)

    return t

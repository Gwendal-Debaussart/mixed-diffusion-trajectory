import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from visualization.style import get_color_method
except ModuleNotFoundError:
    from style import get_color_method


DEFAULT_INPUT = "tables/runtime_scaling/runtime_scaling.csv"
DEFAULT_OUTPUT_DIR = "figures/runtime_scaling"


def _method_visual(method: str):
    """
    Return (label, color, linestyle) for runtime methods.
    """
    mapping = {
        "random_mdt": ("Random MDT", "Random MDT", "-"),
        "random_mdt_with_selection": (
            "Random MDT (time selection)",
            "Random MDT",
            "--",
        ),
        "multiview_diffusion": (
            "Multi-view Diffusion Maps",
            "Multi-view Diffusion Maps",
            "-",
        ),
        "integrated_diffusion": (
            "Integrated Diffusion Maps",
            "Integrated Diffusion Maps",
            "-",
        ),
        "alternating_diffusion": ("Alternating Diffusion", "Alternating Diffusion", "-"),
        "mdt_direct": ("Direct MDT", "Direct MDT", "-"),
        "mdt_contrastive": ("Contrastive MDT", "Direct MDT", "-"),
    }

    label, color_key, linestyle = mapping.get(method, (method, method, "-"))
    return label, get_color_method(color_key), linestyle


def _prepare_dataframe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Runtime CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"n_samples", "method", "total_seconds_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Coerce numeric fields so plotting ignores malformed values safely.
    numeric_cols = [
        c
        for c in [
            "n_samples",
            "selection_seconds_mean",
            "operator_seconds_mean",
            "embedding_seconds_mean",
            "kmeans_seconds_mean",
            "total_seconds_mean",
        ]
        if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["n_samples", "total_seconds_mean"])
    df = df.sort_values(["method", "n_samples"]).reset_index(drop=True)
    return df


def _plot_total_runtime(df: pd.DataFrame, output_dir: str, logy: bool = False) -> None:
    plt.figure()

    for method, grp in df.groupby("method"):
        x = grp["n_samples"].to_numpy()
        y = grp["total_seconds_mean"].to_numpy()
        label, color, linestyle = _method_visual(str(method))
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=color,
            linestyle=linestyle,
            label=label,
        )

    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Runtime (s)")
    if logy:
        plt.yscale("log")
        plt.ylabel("Mean Runtime (s, log scale)")

    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    suffix = "_log" if logy else ""
    plt.savefig(os.path.join(output_dir, f"runtime_total{suffix}.pdf"))
    plt.close()


def _plot_components(df: pd.DataFrame, output_dir: str) -> None:
    component_cols = [
        "selection_seconds_mean",
        "operator_seconds_mean",
        "embedding_seconds_mean",
        "kmeans_seconds_mean",
    ]
    component_cols = [c for c in component_cols if c in df.columns]
    if not component_cols:
        return

    methods = list(df["method"].dropna().unique())
    n_methods = len(methods)

    n_cols = 2
    n_rows = int(np.ceil(n_methods / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.7 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for i, method in enumerate(methods):
        ax = axes_flat[i]
        grp = df[df["method"] == method].sort_values("n_samples")
        x = grp["n_samples"].to_numpy()

        for col in component_cols:
            y = grp[col].to_numpy()
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=col.replace("_seconds_mean", ""))

        ax.set_title(method)
        ax.set_xlabel("n_samples")
        ax.set_ylabel("Mean Runtime (s)")
        ax.grid(alpha=0.25)

    for j in range(n_methods, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "runtime_components_by_method.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_runtimes(csv_path: str, output_dir: str, logy: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    df = _prepare_dataframe(csv_path)
    _plot_total_runtime(df, output_dir, logy=False)
    if logy:
        _plot_total_runtime(df, output_dir, logy=True)
    _plot_components(df, output_dir)

    print(f"Saved plots to: {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot runtime scaling benchmark results.")
    parser.add_argument(
        "--csv",
        default=DEFAULT_INPUT,
        help=f"Path to runtime CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Also save a log-y total runtime plot.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    plot_runtimes(csv_path=args.csv, output_dir=args.outdir, logy=args.logy)


if __name__ == "__main__":
    main()

import argparse
import os
import sys

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import pandas as pd


def _reduced_name(method_name: str) -> str:
    """
    Reduce method names for compact table display.
    """
    reductions = {
        "Alternating Diffusion": "AD",
        "Integrated Diffusion Maps": "ID",
        "Multi-view Diffusion Maps": "MVD",
        "Direct MDT": "MDT-chs",
        "Composite Diffusion Maps": "ComD",
        "Cross Diffusion Maps": "CrD",
        "Powered Alternating Diffusion": "p-AD",
        "Random Convex MDT": "MDT-CVX-Rand",
        "Random MDT": "MDT-Rand",
        "Beam-Search MDT": "MDT-Bsc",
        "Single-view Diffusion Maps (view 1)": "DM (V1)",
        "Single-view Diffusion Maps (view 2)": "DM (V2)",
        "Single-view Diffusion Maps (view 3)": "DM (V3)",
        "Single-view Diffusion Maps (view 4)": "DM (V4)",
        "Single-view Diffusion Maps (view 5)": "DM (V5)",
        "Single-view Diffusion Maps (view 6)": "DM (V6)",
    }
    return reductions.get(method_name, method_name)


DEFAULT_INPUT = "tables/runtime_benchmark/runtime_dataset.csv"
DEFAULT_OUTPUT = "tables/runtime_benchmark/runtime_dataset.tex"


def _dataset_label(row: pd.Series, include_noise: bool) -> str:
    name = str(row["dataset"])
    if include_noise and "noise_factor" in row.index and pd.notna(row["noise_factor"]):
        return f"{name} (noise={float(row['noise_factor']):.2f})"
    return name


def _format_value(mean_v: float, std_v: float | None, decimals: int, include_std: bool) -> str:
    if not np.isfinite(mean_v):
        return "-"
    if include_std and std_v is not None and np.isfinite(std_v):
        return f"{mean_v:.{decimals}f} $\\pm$ {std_v:.{decimals}f}"
    return f"{mean_v:.{decimals}f}"


def build_runtime_latex_table(
    csv_path: str,
    output_path: str,
    value_col: str,
    std_col: str | None,
    decimals: int,
    caption: str,
    label: str,
    include_noise: bool,
    reduce_method_names: bool,
) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Runtime CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"dataset", "method", value_col}
    missing = sorted(required_cols.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if std_col is not None and std_col not in df.columns:
        raise ValueError(f"Requested std column not found: {std_col}")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if std_col is not None:
        df[std_col] = pd.to_numeric(df[std_col], errors="coerce")

    df = df.dropna(subset=["dataset", "method", value_col])
    if df.empty:
        raise ValueError("Runtime CSV has no valid rows after filtering.")

    df["dataset_label"] = df.apply(lambda row: _dataset_label(row, include_noise), axis=1)
    if reduce_method_names:
        df["method_label"] = df["method"].astype(str).map(_reduced_name)
    else:
        df["method_label"] = df["method"].astype(str)

    dataset_order = list(df["dataset_label"].drop_duplicates())
    method_order = list(df["method_label"].drop_duplicates())

    table = pd.DataFrame(index=method_order, columns=dataset_order, dtype=object)

    for dataset_label in dataset_order:
        sub = df[df["dataset_label"] == dataset_label].copy()
        if sub.empty:
            continue

        best_idx = sub[value_col].idxmin()
        best_method = sub.loc[best_idx, "method_label"]

        for _, row in sub.iterrows():
            method_label = row["method_label"]
            mean_v = float(row[value_col])
            std_v = float(row[std_col]) if std_col is not None and pd.notna(row[std_col]) else None
            value_str = _format_value(mean_v, std_v, decimals, include_std=std_col is not None)
            if method_label == best_method:
                value_str = f"\\textbf{{{value_str}}}"
            table.loc[method_label, dataset_label] = value_str

    table = table.fillna("-")
    table.index.name = "Method"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_cols = len(table.columns) + 1
    col_spec = "l" * n_cols
    header_row = " & ".join([""] + [str(col) for col in table.columns]) + r" \\"
    method_header = "Method" + " & " * len(table.columns) + r" \\"

    body_rows = []
    for method_name, row in table.iterrows():
        values = [str(row[col]) for col in table.columns]
        body_rows.append(" & ".join([str(method_name)] + values) + " \\\\")

    latex_lines = [
        r"\begin{table}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row,
        method_header,
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex = "\n".join(latex_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"LaTeX table written to: {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export runtime benchmark CSV to a LaTeX table.")
    parser.add_argument("--csv", default=DEFAULT_INPUT, help=f"Input runtime CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output .tex path (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--value-col",
        default="total_seconds_mean",
        help="Column to tabulate (default: total_seconds_mean)",
    )
    parser.add_argument(
        "--std-col",
        default="total_seconds_std",
        help="Optional std column for mean±std formatting; use empty string to disable.",
    )
    parser.add_argument("--decimals", type=int, default=3, help="Number of decimals (default: 3)")
    parser.add_argument(
        "--caption",
        default="Runtime benchmark across datasets (lower is better).",
        help="LaTeX table caption",
    )
    parser.add_argument("--label", default="tab:runtime_dataset", help="LaTeX table label")
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Append noise level to dataset names when available.",
    )
    parser.add_argument(
        "--reduce-method-names",
        action="store_true",
        help="Use compact aliases (e.g., AD, ID) instead of canonical method names.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    std_col = args.std_col if args.std_col.strip() else None

    build_runtime_latex_table(
        csv_path=args.csv,
        output_path=args.out,
        value_col=args.value_col,
        std_col=std_col,
        decimals=args.decimals,
        caption=args.caption,
        label=args.label,
        include_noise=args.include_noise,
        reduce_method_names=args.reduce_method_names,
    )


if __name__ == "__main__":
    main()

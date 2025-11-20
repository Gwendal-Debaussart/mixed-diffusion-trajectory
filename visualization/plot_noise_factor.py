from .style import get_color_method, get_marker_list, reduced_name
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_noise_factor():
    noise_values = np.arange(0.05, 0.75, 0.05)
    variants = ["lindenbaum", "kuchroo"]

    os.makedirs("figures", exist_ok=True)

    all_dfs = []
    for variant in variants:
        for nf in noise_values:
            path = f"tables/benchmark_results/mnist_{variant}_noise_{nf:.2f}.csv"
            if not os.path.exists(path):
                print(f"[WARNING] missing file : {path} (skipping)")
                continue
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"[ERROR] unable to read {path} : {e}")
                continue

            if "metric" not in df.columns:
                print(f"[WARNING] 'metric' absent in {path} — file ignored")
                continue

            df = df[df["metric"] == "ami"].copy()
            if df.empty:
                continue

            df["noise_factor"] = nf
            df["variant"] = variant
            all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(
            "No data loaded: check paths / CSV files."
        )

    combined = pd.concat(all_dfs, ignore_index=True)

    for col in ["method", "mean", "noise_factor", "variant"]:
        if col not in combined.columns:
            raise RuntimeError(f"Expected column missing in CSVs: '{col}'")

    combined["noise_factor"] = combined["noise_factor"].astype(float)
    combined = combined.sort_values(["variant", "method", "noise_factor"])

    methods = combined["method"].unique().tolist()
    methods = [
        m for m in methods if m not in ["Affinity Addition", "Affinity Multiplication"]
    ]
    methods.sort()
    markers = get_marker_list()
    linestyles = ["-", "--", "-.", ":"]
    FIGSIZE = (9, 6)

    RIGHT = 0.78
    mdt_methods = [m for m in methods if "MDT" in m]
    other_methods = [m for m in methods if "MDT" not in m]

    ordered_methods = mdt_methods + other_methods
    for i, variant in enumerate(variants):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for k, method in enumerate(ordered_methods):
            df_m = combined[
                (combined["method"] == method) & (combined["variant"] == variant)
            ]
            if df_m.empty:
                continue
            df_m = df_m.sort_values("noise_factor")
            z = 10 if "MDT" in method else 5
            a = 1 if "MDT" in method else 1
            ax.plot(
                df_m["noise_factor"].values,
                df_m["mean"].values,
                linewidth=1,
                label=reduced_name(method),
                linestyle=linestyles[k % len(linestyles)],
                marker=markers[k % len(markers)],
                color=get_color_method(method, color_scheme="mdt-focus"),
                zorder=z,
                alpha=a,
            )

        ax.set_xlabel("Noise Factor")
        ax.set_ylabel("AMI")
        fig.subplots_adjust(right=RIGHT)

        handles, labels = ax.get_legend_handles_labels()
        if i == 1 and handles:
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
        else:
            pass

        outpath = f"figures/noise_factor_{variant}.pdf"
        try:
            fig.savefig(outpath)
            print(f"[OK] PDF created: {outpath}")
        except Exception as e:
            print(f"[ERROR] unable to save {outpath} : {e}")
        finally:
            plt.close(fig)
#!/usr/bin/env python3
"""
Enumerate one-hot MDT trajectories up to a given length for a dataset,
compute MDT operator for each trajectory and evaluate CHS and AMI.

Usage:
  python experiments/exhaustive_tree.py --dataset <name> --max-length 3 --out results.csv

This writes a CSV with columns: dataset, noise_factor, length, trajectory, chs, ami
"""
import os
import sys

if __package__ in (None, ""):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Configuration: edit these values directly instead of using a CLI parser
DATASET = "isolet_lindenbaum"
MAX_LENGTH = 4
NOISE_FACTOR = None
OUT = "tables/exhaustive_trajectories.csv"
LIMIT = 100000

import itertools
import numpy as np
import pandas as pd

from benchmarks.load_dataset import load_preprocessed_dataset, get_num_clusters
from mdt.mdt_utils import mdt_operator
from utilities.evaluate import get_clustering, evaluate_labels


def enumerate_one_hot_trajectories(k, length):
    # yields arrays shape (length, k)
    eye = np.eye(k, dtype=float)
    for seq in itertools.product(range(k), repeat=length):
        yield eye[list(seq), :]


def trajectory_id(seq, k):
    """Map a trajectory sequence to a contiguous integer ID.

    For two views this gives:
      length 1 -> 0, 1
      length 2 -> 2, 3, 4, 5
      length 3 -> 6 .. 13
    """
    length = len(seq)
    offset = sum(k ** i for i in range(1, length))
    value = 0
    for item in seq:
        value = value * k + int(item)
    return offset + value


def main():
    # Use top-level configuration values defined above
    dataset = DATASET
    nf = NOISE_FACTOR
    args_max_length = MAX_LENGTH
    out_path = OUT
    safety_limit = LIMIT

    # Load dataset and ensure we get (X_preprocessed, X_views, Y)
    def load_with_views(name, noise_factor=None):
        if noise_factor is not None:
            loaded = load_preprocessed_dataset(name, return_views=True, noise_factor=noise_factor)
        else:
            loaded = load_preprocessed_dataset(name, return_views=True)
        if len(loaded) == 2:
            raise RuntimeError("load_preprocessed_dataset did not return views; call with return_views=True")
        X_pre, X_views, Y = loaded
        return X_pre, X_views, Y

    X_pre, X_views, Y = load_with_views(dataset, nf)

    k = len(X_pre)
    num_clusters = get_num_clusters(dataset)

    rows = []
    total = 0
    for L in range(1, args_max_length + 1):
        count_this_len = k ** L
        if total + count_this_len > safety_limit:
            print(f"Reached safety limit at total={total}; skipping remaining lengths")
            break
        print(f"Enumerating length={L} (count={count_this_len})")
        for traj in enumerate_one_hot_trajectories(k, L):
            total += 1
            seq = np.argmax(traj, axis=1).tolist()
            # compute operator
            try:
                W = mdt_operator(traj, X_pre)
            except Exception as e:
                print(f"Failed to build operator for traj {traj}: {e}")
                continue

            emb = None
            try:
                # use default SVD embedding with n_components = num_clusters
                from utilities.evaluate import get_embedding

                emb = get_embedding(W, n_components=num_clusters, method="svd")
            except Exception:
                # fallback: use operator directly (some methods expect embeddings)
                emb = W

            labels = get_clustering(emb, num_clusters)
            scores = evaluate_labels(Y, X_pre, labels, metric=["chs", "ami"]) or {}

            chs = scores["chs"] if isinstance(scores, dict) and "chs" in scores else float("nan")
            ami = scores["ami"] if isinstance(scores, dict) and "ami" in scores else float("nan")

            rows.append({
                "dataset": dataset,
                "noise_factor": nf,
                "length": L,
                "trajectory": trajectory_id(seq, k),
                "chs": chs,
                "ami": ami,
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

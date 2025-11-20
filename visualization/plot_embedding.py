import matplotlib.pyplot as plt
import os
from .style import *

def plot_embedding(embedding, Y, type: str, name=None, save_dir="figures/embeddings", show=True):
    """
    Plots the 2D embedding with different styles based on the type.
    Parameters:
    -----------
    embedding : np.ndarray
        The 2D embedding to plot.
    Y : np.ndarray
        The labels or values associated with the data points.
    type : str
        The type of plot to create ('clustering' or 'manifold').
    save_path : str, optional
        The path to save the plot. Defaults to "figure/embeddings".
    """
    col_list = get_col_list()
    mark_list = get_marker_list()

    if type == "clustering":
        K = max(Y) + 1
        plt.figure()
        for k in range(K):
            plt.scatter(
                embedding[Y == k, 0],
                embedding[Y == k, 1],
                marker=mark_list[k],
                color=col_list[k],
                label="Cluster " + str(k),
            )
            plt.legend(loc="right", bbox_to_anchor=(1.25, 0.5))
    elif type == "manifold":
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=Y, cmap=get_cmap("two_tone"))
    else:
        raise ValueError("Type must be 'clustering' or 'manifold'.")
    if name is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{name}.pdf", format="pdf", bbox_inches="tight")
    if show:
        plt.show()
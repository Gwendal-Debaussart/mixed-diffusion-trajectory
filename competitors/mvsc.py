import numpy as np
from mvlearn.cluster import MultiviewSpectralClustering

def mvsc_embedding(views, n_clusters, **kwargs):
    """
    Parameters
    ----------
    views: list of arrays [(n_samples, d1), (n_samples, d2), ...]
    n_clusters: int
        The number of clusters to form.

    Returns
    -------
    embedding: array-like, shape (n_samples, n_clusters)
        The embedding of the data.
    """
    mvsc = MultiviewSpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors', # We use nearest_neighbors affinity, because MVSC uses ARPACK, which does not support dense matrices. This causes issues in the context of higher chosen $t$ for the MVSC + MDT method, where the operator becomes dense.
    )
    mvsc.fit(views)
    embedding = mvsc.embedding_

    if isinstance(embedding, (list, tuple)):
        if len(embedding) == 1:
            embedding = embedding[0]
        else:
            embedding = np.mean(np.stack(embedding, axis=0), axis=0)

    embedding = np.asarray(embedding)
    if embedding.ndim == 3:
        embedding = np.mean(embedding, axis=0)

    return embedding
import numpy as np
from mvlearn.embed import GCCA


def _to_2d_embedding(embedding):
    """
    Convert GCCA output to a 2D (n_samples, n_components) embedding.
    """
    if isinstance(embedding, (list, tuple)):
        if len(embedding) == 1:
            return np.asarray(embedding[0])
        return np.mean(np.stack([np.asarray(e) for e in embedding], axis=0), axis=0)

    arr = np.asarray(embedding)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # mvlearn may return one embedding per view as (n_views, n_samples, n_components)
        return np.mean(arr, axis=0)

    raise ValueError(f"Unexpected GCCA embedding shape: {arr.shape}")

def gcca_embedding(views, n_components):
    """
    Compute GCCA embedding from multiple views, using mvlearn's implementation.

    The embedding is averaged across views, as GCCA returns one embedding per view.

    Parameters:
    ----------
    views: list of np.ndarray
        List of input views (kernel matrices).
    n_components: int
        Desired dimensionality of the embedding.
    Returns:
    -------
    embedding: np.ndarray
        The GCCA embedding of the data.
    """
    min_rank = min(np.linalg.matrix_rank(v) for v in views)
    effective_components = max(1, min(int(n_components), int(min_rank)))

    gcca = GCCA(n_components=effective_components)

    embedding = gcca.fit_transform(views)
    return _to_2d_embedding(embedding)
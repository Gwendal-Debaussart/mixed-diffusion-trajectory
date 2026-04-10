import numpy as np
from mvlearn.embed import GCCA

def gcca(views, n_components):
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

    if isinstance(embedding, (list, tuple)):
        if len(embedding) == 1:
            return embedding[0]
        return np.mean(np.stack(embedding, axis=0), axis=0)

    return embedding
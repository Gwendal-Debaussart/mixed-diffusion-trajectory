import numpy as np
from utilities.evaluate import get_embedding

def method_to_embedding(operator, X_preprocessed, method, dim_embedd=10):
    """
    Convert the operator to an embedding depending on the method type.
    """
    decomp = method.get("decomp_method", "svd")
    embedding = get_embedding(operator, dim_embedd, decomp)

    if isinstance(operator, np.ndarray):
        n_samples = len(X_preprocessed)
        if operator.shape[0] > n_samples:
            operator = operator[:n_samples, :n_samples]
    return embedding
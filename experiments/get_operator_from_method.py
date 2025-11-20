from benchmarks.utilities import get_kernel_matrix

def get_operator_from_method(method, dataset_name, X_preprocessed, X_views):
    """
    Compute an embedding (or operator) for a given method and dataset.
    Parameters:
    - method: dict defining the method to use (see 'methods' structure)
    - dataset_name: str, name of the dataset
    - X_preprocessed: list of preprocessed data matrices (one per view)
    - X_views: list of original data matrices (one per view)
    Returns:
    - operator(s) computed by the method
    Notes:
    - method['params'] can be a callable: params(dataset_name) -> dict
    - If 'params' is not defined, defaults to empty dict.
    - Handles 'multi_view', 'single_view', and 'default' types.
    - Truncates operator if its shape exceeds the dataset (for MVDM).
    """
    params_callable = method.get("params", lambda dn: {})
    X_kernels = [get_kernel_matrix(Xv, normalize=False) for Xv in X_views]
    params = params_callable(dataset_name)

    if method["input_type"] == "kernels":
        return method["func"](X_kernels, **params)
    if method["input_type"] == "views":
        return method["func"](X_views, **params)
    else:
        return method["func"](X_preprocessed, **params)
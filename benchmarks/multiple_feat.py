import numpy as np
import os
import pandas as pd
import sklearn.preprocessing as pre


def multiple_feat():
    """
    None of the parameters are used, they are only here for consistency reasons.
    """
    list_views = []
    true_labels = np.array([i // 200 for i in range(2000)])
    base_dir = os.path.join(os.path.dirname(__file__), "source/multiple+features")
    for feat in ["fac", "kar", "zer", "mor", "pix", "fou"]:
        X_path = os.path.join(
            base_dir, "mfeat-" + feat
        )
        tmp = pd.read_csv(X_path, header=None, sep=r"\s+")
        list_views.append(pre.MinMaxScaler().fit_transform(tmp))
    return list_views, true_labels

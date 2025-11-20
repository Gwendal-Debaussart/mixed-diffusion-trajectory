import os
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from sklearn import preprocessing as pre
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

def olivetti():
    """
    Load or download Olivetti faces dataset and simulate two views:
    - View 1: pixel intensities (scaled 0-1)
    - View 2: HOG features (scaled 0-1)
    Saves/loads dataset from 'source/olivetti.npz'
    """
    save_dir = os.path.join(os.path.dirname(__file__), "source/olivetti")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "olivetti.npz")

    if os.path.exists(save_path):
        data = np.load(save_path)
        X1, X2, Y = data["X1"], data["X2"], data["Y"]
    else:
        print("Downloading Olivetti dataset...")
        faces = fetch_olivetti_faces()
        X1, X2, Y = faces.data, faces.images, faces.target
        np.savez_compressed(save_path, X1=X1, X2=X2, Y=Y)
        print(f"Olivetti dataset saved to {save_path}")
    X1 = pre.MinMaxScaler().fit_transform(X1)
    X2 = np.array([hog(resize(img, (32, 32))) for img in X2])
    X2 = pre.MinMaxScaler().fit_transform(X2)
    X1 = PCA(n_components=150).fit_transform(X1)
    X2 = PCA(n_components=150).fit_transform(X2)
    X1 /= np.linalg.norm(X1, axis=1, keepdims=True) + 1e-9
    X2 /= np.linalg.norm(X2, axis=1, keepdims=True) + 1e-9

    return [X1, X2], Y

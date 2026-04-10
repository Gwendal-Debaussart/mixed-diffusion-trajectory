import numpy as np
import torch
from mdt.mdt_utils import *


def contrastive_mdt_loss(X, W, view_weights=None) -> torch.Tensor:
    """
    Compute the contrastive MDT loss.
    Parameters:
    -----------
    X : list of np.arrays
        List of views, each of shape (n_samples, n_features).
    W : np.array
        Weight matrix of shape (n_samples, n_samples).
    view_weights : list of floats, optional
        Weights for each view. If None, equal weights are used.
    Returns:
    --------
    loss : float
        The computed contrastive MDT loss.
    """
    eps = 1e-12
    W = torch.nan_to_num(W, nan=0.0, posinf=20.0, neginf=-20.0)
    W = torch.clamp(W, min=-20.0, max=20.0)
    exW = torch.exp(W)
    M = exW.clone().fill_diagonal_(0)
    D = M.sum(axis=1).clamp_min(eps)
    idx = [[torch.argwhere(torch.tensor(x)[i, :] > 0).flatten() for i in range(x.shape[0])] for x in X]
    loss = 0
    if view_weights is None:
        view_weights = [1/len(X)] * len(X)
    for v in range(len(X)):
        loss_v = 0
        for i in range(X[0].shape[0]):
            if idx[v][i].numel() == 0:
                continue
            probs = torch.clamp(exW[i, idx[v][i]] / D[i], min=eps)
            loss_v -= torch.sum(torch.log(probs))
        loss += view_weights[v] * loss_v
    return loss / len(X[0])

def mdt_operator_torch(trajectory, X) -> torch.Tensor:
    """
    Constructs the mixed-view diffusion trajectory operator from a given trajectory using PyTorch.
    """
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.as_tensor(trajectory, dtype=torch.float32)

    if isinstance(X, torch.Tensor):
        X_tensor = X.to(dtype=trajectory.dtype, device=trajectory.device)
    else:
        X_tensor = torch.as_tensor(np.asarray(X), dtype=trajectory.dtype, device=trajectory.device)

    if X_tensor.ndim != 3:
        raise ValueError("X must be a tensor/list of shape (n_views, n_samples, n_samples).")
    if trajectory.ndim != 2 or trajectory.shape[1] != X_tensor.shape[0]:
        raise ValueError("trajectory must have shape (t, n_views).")

    n_samples = X_tensor.shape[1]
    if X_tensor.shape[2] != n_samples:
        raise ValueError("Each view operator must be square (n_samples x n_samples).")

    W = torch.eye(n_samples, dtype=trajectory.dtype, device=trajectory.device)

    mixed_ops = torch.einsum("tv,vij->tij", trajectory, X_tensor)
    for M in mixed_ops:
        W = M @ W
    return W

def mdt_contrastive(X, t, view_weights=None) -> np.ndarray:
    """
    Optimize the MDT operator using contrastive loss.
    Parameters:
    -----------
    X : list of np.arrays
        List of views, each of shape (n_samples, n_features).
    t : int
        Number of time steps.
    view_weights : list of floats, optional
        Weights for each view. If None, equal weights are used.
    Returns:
    --------
    W : np.array
        The optimized MDT operator of shape (n_samples, n_samples).
    """
    A = torch.rand(t, len(X), dtype=torch.float32, requires_grad=True)
    X_torch = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    optimizer = torch.optim.Adam([A], lr=0.05)
    best_loss = float("inf")
    best_A = None

    for _ in range(80):
        optimizer.zero_grad()
        A2 = torch.softmax(A, dim=1)
        P = mdt_operator_torch(A2, X_torch)
        if not torch.isfinite(P).all():
            continue
        l: torch.Tensor = contrastive_mdt_loss(X, P, view_weights=view_weights)
        if not torch.isfinite(l):
            continue
        l.backward()
        torch.nn.utils.clip_grad_norm_([A], max_norm=1.0)
        optimizer.step()

        loss_value = float(l.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_A = A.detach().clone()

    if best_A is None:
        best_A = A.detach().clone()

    A2 = torch.softmax(best_A, dim=1).detach().cpu().numpy()
    return mdt_operator(A2, X)

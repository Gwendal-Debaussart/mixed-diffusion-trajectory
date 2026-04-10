import numpy as np
import torch
from mdt.mdt_utils import *


def contrastive_mdt_loss(idx, W, view_weights=None) -> torch.Tensor:
    """
    Compute the contrastive MDT loss — fully vectorized.
    Parameters:
    -----------
    idx : list of BoolTensors of shape (n_samples, n_samples)
        Precomputed boolean masks for positive pairs per view.
    W : torch.Tensor
        Weight matrix of shape (n_samples, n_samples).
    view_weights : list of floats, optional
        Weights for each view. If None, equal weights are used.
    """
    eps = 1e-12
    W = torch.nan_to_num(W, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    exW = torch.exp(W)
    M = exW * (1 - torch.eye(exW.shape[0], device=exW.device, dtype=exW.dtype))
    D = M.sum(dim=1).clamp_min(eps)  # (n,)
    log_probs = torch.log((exW / D.unsqueeze(1)).clamp_min(eps))  # (n, n)

    if view_weights is None:
        view_weights = [1 / len(idx)] * len(idx)

    loss = 0.0
    for v, mask in enumerate(idx):
        loss -= view_weights[v] * log_probs[mask].sum()

    return loss / W.shape[0]


def mdt_operator_torch(trajectory, X):
    """
    Constructs the mixed-view diffusion trajectory operator from a given trajectory.
    Parameters:
    -----------
    trajectory: torch.Tensor of shape (T, k)
    X: torch.Tensor of shape (k, n, n) or list of np.ndarray of shape (n, n)
    Returns:
    --------
    W: torch.Tensor of shape (n, n)
    """
    if torch.is_tensor(X):
        X_stack = X.to(dtype=torch.float64)
        if X_stack.ndim != 3:
            raise ValueError("X tensor must have shape (num_views, n, n).")
    else:
        X_stack = torch.stack(
            [torch.as_tensor(x, dtype=torch.float64) for x in X], dim=0
        )

    if not torch.is_tensor(trajectory):
        trajectory = torch.as_tensor(trajectory, dtype=torch.float64)
    else:
        trajectory = trajectory.to(dtype=torch.float64)

    weighted = torch.einsum("tk,knm->tnm", trajectory, X_stack)
    return matrix_chain_product_torch(weighted)


def matrix_chain_product_torch(matrices):
    while len(matrices) > 1:
        if len(matrices) % 2 == 1:
            leftover = matrices[-1:]
            matrices = matrices[:-1]
        else:
            leftover = None
        pairs = torch.matmul(matrices[1::2], matrices[::2])
        if leftover is not None:
            matrices = torch.cat([pairs, leftover])
        else:
            matrices = pairs
    return matrices[0]


def precompute_idx(X, device="cpu"):
    """
    Precompute boolean masks for positive pairs per view.
    Call once before the optimization loop.
    """
    return [torch.as_tensor(x > 0, dtype=torch.bool, device=device) for x in X]


def mdt_contrastive(X, t, view_weights=None) -> np.ndarray:
    """
    Optimize the MDT operator using contrastive loss.
    Parameters:
    -----------
    X : list of np.arrays of shape (n_samples, n_features)
    t : int — number of time steps
    view_weights : list of floats, optional
    Returns:
    --------
    W : np.array of shape (n_samples, n_samples)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A = torch.rand(t, len(X), dtype=torch.float32, requires_grad=True, device=device)
    X_torch = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    idx = precompute_idx(X, device=device)

    optimizer = torch.optim.Adam([A], lr=0.05)
    best_loss = float("inf")
    best_A = None

    for step in range(80):
        optimizer.zero_grad()
        A2 = torch.softmax(A, dim=1)
        P = mdt_operator_torch(A2, X_torch)

        if not torch.isfinite(P).all():
            continue

        l = contrastive_mdt_loss(idx, P, view_weights=view_weights)

        if not torch.isfinite(l):
            continue

        l.backward()
        if step % 10 == 0:
            print(f"Step {step:3d} | loss: {l.item():.4f}")
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

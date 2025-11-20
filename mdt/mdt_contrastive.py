import numpy as np
import torch
from mdt.mdt_utils import *


def contrastive_mdt_loss(X, W, view_weights=None):
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
    exW = torch.exp(W)
    M = exW.clone().fill_diagonal_(0)
    D  = M.sum(axis=1)
    idx = [[torch.argwhere(torch.tensor(x)[i,:] > 0) for i in range(x.shape[0])] for x in X]
    loss = 0
    if view_weights is None:
        view_weights = [1/len(X)] * len(X)
    for v in range(len(X)):
        loss_v = 0
        for i in range(X[0].shape[0]):
            loss_v -= torch.sum(torch.log(exW[i, idx[v][i]] / D[i]))
        loss += view_weights[v] * loss_v
    return loss / len(X[0])

def mdt_operator_torch(trajectory, X):
    """
    Constructs the mixed-view diffusion trajectory operator from a given trajectory using PyTorch.
    """
    t, _ = trajectory.shape
    W = torch.eye(X[0].shape[0])
    for i in range(t):
        W =  sum(a * torch.tensor(P, dtype=torch.float32) for a, P in zip(trajectory[i], X)) @ W
    return W

def mdt_contrastive(X, t, view_weights=None):
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
    A = np.random.rand(t, len(X))
    A = torch.tensor(A, requires_grad=True)
    optimizer = torch.optim.Adam([A], lr=0.8)
    l_min = None
    for _ in range(40):
      optimizer.zero_grad()
      A2 = A / A.sum(1).reshape(-1, 1)
      P = mdt_operator_torch(A2, X)
      l = contrastive_mdt_loss(X, P, view_weights = view_weights)
      l.backward()
      print("l", l)
      P = P.detach()
      optimizer.step()
      if l_min == None:
        l_min = l
        A_min = A.detach()
      elif l_min < l:
        l_min = l
        A_min = A.detach()

    A2 = A_min / A_min.sum(1).reshape(-1, 1)

    return mdt_operator(np.array(A2), X)

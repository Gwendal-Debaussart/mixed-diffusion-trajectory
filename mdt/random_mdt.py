import numpy as np
from .mdt_utils import mdt_operator


def random_mdt(X, t, convex= False, distribution="dirichlet"):
  """
  Generate a random mixed-view diffusion trajectory (R-MDT) of length t, from the given multi-view data.

  Parameters:
  ---------
  X : list of np.ndarray
      List of transition matrices representing different views of the data.
  t : int
      Length of the trajectory.
  convex : bool, optional
      If True, generates a convex combination trajectory; otherwise, generates a one-hot trajectory. Default is False.
  Returns:
  -------
  trajectory : np.ndarray
      Generated random trajectory of shape (t, len(X)).
  """
  if distribution == "pseudo-uniform":
      if convex:
        trajectory = np.random.rand(t, len(X))
        if t > 1:
          return trajectory / np.sum(trajectory, axis=1, keepdims=True)
        else:
          return trajectory / np.sum(trajectory)

      else:
        trajectory = np.zeros((t, len(X)), dtype=int)
        col_indices = np.random.randint(0, len(X), size=t)
        trajectory[np.arange(t), col_indices] = 1
        return trajectory
  elif distribution == "dirichlet":
      if convex:
        trajectory = np.random.dirichlet(alpha=np.ones(len(X)), size=t)
        return trajectory
      else:
        trajectory = np.zeros((t, len(X)), dtype=int)
        col_indices = np.random.randint(0, len(X), size=t)
        trajectory[np.arange(t), col_indices] = 1
        return trajectory
  else:
      raise ValueError(f"Unsupported distribution type: {distribution}")


def random_mdt_operator(X, t, convex= False, distribution="dirichlet"):
    """
    Generate the MDT operator corresponding to a random mixed-view diffusion trajectory (R-MDT).

    Parameters:
    ---------
    X : list of np.ndarray
        List of transition matrices representing different views of the data.
    t : int
        Length of the trajectory.
    convex : bool, optional
        If True, generates a convex combination trajectory; otherwise, generates a one-hot trajectory. Default is False.
    Returns:
    -------
    W : np.ndarray
        MDT operator generated from the random trajectory.
    """
    trajectory = random_mdt(X, t, convex, distribution)

    return mdt_operator(trajectory, X)
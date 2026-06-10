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


def circulant_mdt_trajectory(X, start=0, shuffle=False, random_state=None):
    """
    Generate a circulant mixed-view diffusion trajectory.

    The trajectory contains each operator exactly once in a single cycle.
    By default it is a cyclic shift of the identity matrix.

    Parameters:
    -----------
    X : list of np.ndarray
        List of transition matrices representing different views of the data.
    start : int, optional
        Starting index for the cyclic shift. Default is 0.
    shuffle : bool, optional
        Whether to randomly shuffle the order of the operators. Default is False.
    random_state : int or None, optional
        Random seed for shuffling. Default is None.

    Returns:
    --------
    trajectory : np.ndarray
        Generated circulant trajectory of shape (len(X), len(X)).
    """
    n_views = len(X)
    if n_views == 0:
        raise ValueError("X must contain at least one operator.")

    if shuffle:
        rng = np.random.default_rng(random_state)
        order = rng.permutation(n_views)
    else:
        order = np.roll(np.arange(n_views), -int(start) % n_views)

    trajectory = np.eye(n_views, dtype=int)[order]
    return trajectory


def circulant_mdt_operator(
    X,
    power=None,
    max_t=50,
    start=0,
    shuffle=False,
    random_state=None,
    cache_key=None,
    return_power=False,
):
    """
    Build the MDT operator from a circulant trajectory and optionally power it.

    If `power` is not provided, the power is selected from the operator's
    singular-value criterion via the existing diffusion-time heuristic.

    Parameters:
    -----------
    X : list of np.ndarray
        List of transition matrices representing different views of the data.
    power : int or None, optional
        Power to which the operator is raised. If None, it is determined by the diffusion-time heuristic. Default is None.
    max_t : int, optional
        Maximum power to consider when determining the diffusion time. Default is 50.
    start : int, optional
        Starting index for the circulant trajectory. Default is 0.
    shuffle : bool, optional
        Whether to randomly shuffle the order of the operators in the trajectory. Default is False.
    random_state : int or None, optional
        Random seed for shuffling. Default is None.
    cache_key : str or None, optional
        Key for caching the diffusion time. Default is None.
    return_power : bool, optional
        Whether to return the selected power along with the operator. Default is False.

    Returns:
    --------
    W : np.ndarray
        The MDT operator raised to the specified power.
    power (optional) : int
        The power to which the operator was raised, returned if `return_power` is True.
    """
    trajectory = circulant_mdt_trajectory(
        X,
        start=start,
        shuffle=shuffle,
        random_state=random_state,
    )
    operator = mdt_operator(trajectory, X)

    if power is None:
        from experiment_utils.get_diffusion_time import get_diffusion_time

        power = get_diffusion_time(operator=operator, max_t=max_t, cache_key=cache_key)

    powered_operator = np.linalg.matrix_power(operator, int(power))
    if return_power:
        return powered_operator, int(power)
    return powered_operator
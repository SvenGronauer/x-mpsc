r"""

Author: Sven Gronauer (sven.gronauer@tum.de)
Created: 29.08.2022
"""
import abc
from typing import Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import gymnasium as gym
import casadi as cs
import torch as th
import torch.nn as nn


def build_mlp_network(
        sizes,
        activation: nn.Module = nn.Tanh,
        output_activation: nn.Module = nn.Identity,
):
    layers = list()
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        affine_layer = nn.Linear(sizes[j], sizes[j+1])
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


def casadi_add_two_ellipsoids(
        c1: Union[cs.DM, cs.MX, cs.SX],
        c2: Union[cs.DM, cs.MX, cs.SX],
        Q1: Union[cs.DM, cs.MX, cs.SX],
        Q2: Union[cs.DM, cs.MX, cs.SX]
) -> Tuple[Union[cs.DM, cs.MX, cs.SX], Union[cs.DM, cs.MX, cs.SX]]:
    r"""Over-approximate the sum of two n-dimensional ellipsoidal sets.

    Parameters
    ----------
    c1: n x 1 vector
        Center of first ellipsoid
    c2: n x 1 vector
        Center of second ellipsoid
    Q1: n x n matrix
        Shape of first ellipsoid
    Q2: n x n matrix
        Shape of second ellipsoid

    Returns
    -------
    c_new: n x 1 array
        Center of the new ellipsoid
    Q_new: n x n array
        Shape matrix of the new ellipsoid
    """
    alpha = cs.sqrt(cs.trace(Q1) / cs.trace(Q2))
    c_new = c1 + c2
    Q_new = (1 + (1. / alpha)) * Q1 + (1 + alpha) * Q2
    return c_new, Q_new


def casadi_ellipsoid_in_polytope_constraint(
        c: Union[cs.DM, cs.MX, cs.SX],
        Q: Union[cs.DM, cs.MX, cs.SX],
        H: Union[cs.DM, cs.MX, cs.SX],
        d: Union[cs.DM, cs.MX, cs.SX],
) -> Union[cs.DM, cs.MX, cs.SX]:
    r"""Return the constraints: r + Hc -d <= 0.

    Parameters
    ----------
    c:  n x 1 vector
        Center of ellipsoid
    Q:  n x n matrix
        Shape of first ellipsoid
    H:  n x n matrix
        Constraint matrix of polytope
    d

    Returns
    -------
    s:  n x 1 vector
        distance to polytope boundary
    """
    Hc = cs.mtimes(H, c)
    # sum1 is row-wise addition of elements
    row_vector = cs.sum1(H.T * cs.mtimes(Q, H.T))
    r = cs.sqrt(row_vector).T
    return r + Hc - d


def minkowski_difference_ellipsoids(
        # c1: Union[cs.DM, cs.MX, cs.SX],
        Q1: Union[cs.DM, cs.MX, cs.SX],
        # c2: Union[cs.DM, cs.MX, cs.SX],
        Q2: Union[cs.DM, cs.MX, cs.SX],
# ) -> Tuple[Union[cs.DM, cs.MX, cs.SX], Union[cs.DM, cs.MX, cs.SX]]:
) -> Union[cs.DM, cs.MX, cs.SX]:
    r"""Internal approximation of Minkowski difference of two ellipsoids."""
    p = cs.sqrt(cs.trace(Q1) / cs.trace(Q2))
    # return c1-c2, (1 - 1. / p) * Q1 + (1 - p) * Q2
    return (1 - 1. / p) * Q1 + (1 - p) * Q2


def ellipsoid_constraint(
        x: Union[cs.DM, cs.MX, cs.SX],
        c: Union[cs.DM, cs.MX, cs.SX],
        Q: Union[cs.DM, cs.MX, cs.SX],
) -> Union[cs.DM, cs.MX, cs.SX]:
    r"""Return an ellipsoidal constraint: (x-c)^T Q (x-c)

    Note: it is Q not Q^{-1} to maintain analytical expression

    Usage: opti.subject_to(ellipsoid_constraint(x, c, Q) <= 0)
    """
    #fixme: cs.inv() breaks differentiability!
    return (x - c).T @ cs.inv(Q) @ (x - c) - 1


def get_reduced_box_from_ellipsoid(
        c: Union[cs.DM, cs.MX, cs.SX],
        Q: Union[cs.DM, cs.MX, cs.SX],
        H: Union[cs.DM, cs.MX, cs.SX],
        d: Union[cs.DM, cs.MX, cs.SX],
) -> Tuple[Union[cs.DM, cs.MX, cs.SX], Union[cs.DM, cs.MX, cs.SX]]:
    r"""Reduced box of size: d - sqrt(h^T Q h)

    Note: return tuple can be used as a constraint via:
        Hc <= d - sqrt(h^T Q h)

    Parameters
    ----------
    c:  n x 1 vector
        Center of ellipsoid
    Q:  n x n matrix
        Shape matrix of ellipsoid
    H:  n x n matrix
        Constraint matrix of box
    d

    Returns
    -------
    s:  n x 1 vector
        distance to polytope boundary
    """
    row_vector = cs.sum1(H.T * cs.mtimes(Q, H.T))
    r = cs.sqrt(row_vector).T
    d_reduced = d - r
    Hc = cs.mtimes(H, c)
    return d_reduced, Hc


def bring_to_matrix(
        vector_or_matrix: np.ndarray
) -> np.ndarray:
    r"""Reshapes an array to a matrix."""
    if vector_or_matrix.ndim == 1:
        return vector_or_matrix.reshape((-1, vector_or_matrix.shape[0]))
    elif vector_or_matrix.ndim == 2:
        return vector_or_matrix
    else:
        raise ValueError("Cannot reshape 3D-Tensor to matrix.")


def find_ellipsoid_around_points(
        x: np.ndarray
) -> np.ndarray:
    """

    Parameters
    ----------
    x:  Data are expected as:  N (#points) x nx (data dim)

    Returns
    -------
    L^{-1} matrix parametrizing ellipsoid via || L^{-1} x || <= 1
    """
    N, nx = x.shape

    L_inv = cp.Variable(shape=(nx, nx))
    # b = cp.Variable(shape=(nx, ))
    objective = cp.log_det(L_inv)

    constraints = [cp.norm(L_inv @ x[i, :], 2) <= 1 for i in range(N)]
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(verbose=False)
    return np.asarray(L_inv.value)

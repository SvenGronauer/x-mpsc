r"""

Author: Sven Gronauer (sven.gronauer@tum.de)
Created: 29.08.2022
"""
import abc
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import casadi as cs

# local imports
# import x_mpsc.common.loggers as loggers
# from x_mpsc.models import MassSpringDamperModel
# from x_mpsc.envs.linear_systems.lq_envs import MassSpringDamperEnv
# from x_mpsc.mpc.solvers.mpsc import LinearModelPredictiveSafetyCertification
# import sys
from matplotlib import patches


def is_pos_def(mat: np.ndarray) -> bool:
    if np.array_equal(mat, mat.T):
        try:
            np.linalg.cholesky(mat)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


class BaseSet(abc.ABC):
    """Base class for shapes like ellipsoids, boxes, etc."""
    @abc.abstractmethod
    def add_set(self, other):
        pass

    @abc.abstractmethod
    def draw(self,  ax: plt.Axes, dims: tuple = (0, 1), **kwargs):
        pass

    @abc.abstractmethod
    def get_symbolic_model(self):
        pass

    @abc.abstractmethod
    def transform(self, **kwargs):
        pass


class BoxSet(BaseSet):
    r"""A box is a closed polytope described by: Ax <= b."""
    def __init__(
            self,
            A: Optional[np.array] = None,   # matrix
            b: Optional[np.ndarray] = None,  # vector
            fromEnv: Optional[gym.Env] = None,
            from_space: Optional[gym.Space] = None,
    ):
        if from_space is not None:  # named constructor
            self._from_space(from_space)
        elif fromEnv is not None:  # named constructor
            self._from_env(fromEnv)
        else:
            self.A = A
            self.b = b
            self.dim = A.shape[0]  # fixme sven: this should be index 1?

        assert (self.A.ndim == 2), "A must be a matrix."
        assert (self.b.ndim == 1), "b must be a vector."

        self.symbolic_function = lambda x: self.A @ x - self.b

    def __str__(self):
        return f"A={self.A} @ x <= {self.b}"

    def _rebuild_system_matrices_from_bounds(self):
        self.A = np.vstack((np.eye(self.dim), -np.eye(self.dim)))
        self.b = np.hstack((self.upper_bounds, -self.lower_bounds))
        # print(self.b)
        # print(self.b.ndim)

    @DeprecationWarning
    def _from_env(self, env: gym.Env):  # a named constructor
        if isinstance(env.observation_space, gym.spaces.Box):
            self.upper_bounds = np.array(env.observation_space.high, ndmin=1)
            self.lower_bounds = np.array(env.observation_space.low, ndmin=1)
            self.dim = self.lower_bounds.shape[0]
            self._rebuild_system_matrices_from_bounds()
        else:
            raise ValueError("Expecting env space to be a box.")

    def _from_space(self, space: gym.Space):  # a named constructor
        if isinstance(space, gym.spaces.Box):
            self.upper_bounds = np.array(space.high, ndmin=1)
            self.lower_bounds = np.array(space.low, ndmin=1)
            self.dim = self.lower_bounds.shape[0]
            self._rebuild_system_matrices_from_bounds()
        else:
            raise ValueError("Expecting env space to be a box.")

    def add_set(self, other):
        if isinstance(other, BoxSet):
            assert (other.lower_bounds.shape == self.lower_bounds.shape)
            self.upper_bounds += other.upper_bounds
            self.lower_bounds -= other.upper_bounds
            self.A = np.vstack((np.eye(self.dim), -np.eye(self.dim)))
            self.b = np.hstack((self.upper_bounds, -self.lower_bounds))
        else:
            raise NotImplementedError

    def draw(self,
             ax: plt.Axes,
             dims: tuple = (0, 1),
             color: str = "blue",
             **kwargs):
        assert (self.b.ndim == 1), 'plots support only vectors for b.'
        assert self.A.shape[0] > 0, 'wrong matrix dimensions.'

        # bottom left corner is:
        b_2 = self.b.shape[0] // 2
        bottom_left_corner = -np.array([self.b[b_2+dims[0]], self.b[b_2+dims[1]]]).flatten()
        upper_right_corner = np.array([self.b[dims[0]], self.b[dims[1]]]).flatten()
        width, height = upper_right_corner - bottom_left_corner
        # print(f"corner: {bottom_left_corner} with: {width} height: {height}")
        # print(f"width: {width} height: {height}")
        # Create a Rectangle patch
        rect = patches.Rectangle(
            bottom_left_corner, width, height, linewidth=1,
            edgecolor=color,  facecolor='none')
        ax.add_patch(rect)

    def get_symbolic_model(self):
        return self.symbolic_function

    def subtract_set(self, other):
        r"""Minkowski/Pontryagin set subtraction"""
        if isinstance(other, EllipsoidalSet):
            num_rows = self.A.shape[0]
            assert(num_rows == self.b.shape[0])
            for i in range(num_rows):
                distance_to_bound = np.sqrt(self.A[i] @ other.Q @ self.A[i].T)
                self.b[i] -= distance_to_bound
        else:
            raise NotImplementedError

    def transform(self, **kwargs):
        raise NotImplementedError


class EllipsoidalSet(BaseSet):
    def __init__(
            self,
            Q,   # shape matrix
            c  # center of ellipsoid as vector
    ):
        r"""Ellipsoids are parametrized as:
                (x-c)^T Q^{-1} (x-c) <= 1
        or equivalently with Q = L L^T:
                \| L^{-1} (x-c) \| <= 1
        """
        self.Q = Q
        self.c = c.copy()

        assert (Q.ndim == 2), "Q must be a matrix."
        assert (c.ndim == 1), "x must be a vector."

        # cs.inv(self.Q)

        # Note:
        #   the inverse breaks the construction of the QP in casadi...
        #   since inverse is no analytic operation!
        self.symbolic_function = lambda x: \
            (x-self.c).T @ np.linalg.inv(self.Q) @ (x-self.c) - 1

    def __repr__(self):
        return f"Ellipsoid with c={self.c}) and Q=\n{self.Q}"

    def add_set(
            self,
            other
    ) -> None:
        r"""Minkowski sum of two ellipsoids."""
        if isinstance(other, EllipsoidalSet):
            c_new = self.c + other.c  # add centers
            alpha = np.sqrt(np.trace(self.Q) / np.trace(other.Q))
            Q_tilde = (1 + 1. / alpha) * self.Q + (1 + alpha) * other.Q
            self.c = c_new
            self.Q = Q_tilde
        else:
            raise NotImplementedError

    def copy(self):
        return EllipsoidalSet(Q=self.Q.copy(), c=self.c.copy())

    def draw(
            self,
            ax: plt.Axes,
            centroid: bool = False,
            dims: tuple = (0, 1),  # dimensions that are plotted
            color: Optional[str] = 'blue',
            linestyle: Optional[str] = '-',
            **kwargs
    ) -> None:
        r"""draw ellipsoid into figure axis."""
        assert len(dims) == 2, f"Only two-dimensional plots are supported."
        d1, d2 = dims
        rows = np.array([[d1, d1], [d2, d2]], dtype=np.intp)
        columns = np.array([[d1, d2], [d1, d2]], dtype=np.intp)
        Q = self.Q[rows, columns]

        c = np.array([self.c[dims[0]], self.c[dims[1]]]).flatten()

        num_circle_points = 100
        angles = np.linspace(0, 2 * np.pi, num_circle_points)
        b = np.array([np.cos(angles), np.sin(angles)])  # points on unit circle
        L_inv = np.linalg.cholesky(np.linalg.inv(Q))
        # Important note: Cholesky returns L and not L^T
        # todo sven: why do I have to use L_inv.T instead of L_inv ??
        ellipse_hull = np.linalg.solve(L_inv.T, b).T  # transpose to get shape (N, nx)
        xs = c + ellipse_hull
        if centroid:
            ax.scatter(c[0], c[1],  marker="*", color=color)
        ax.plot(xs[:, 0], xs[:, 1], linestyle=linestyle, color=color)

    def fit(self, data):
        r"""Expecting data to be of shape: (N, nx)
        with N samples and dimension nx.
        """
        assert self.Q.ndim == data.ndim, "dimension mismatch"
        assert self.c.shape[0] == data.shape[1], "dimension mismatch"

        mu = np.mean(data, axis=0)
        sigma = 2 * np.std(data - mu, axis=0)
        self.Q = np.diag(sigma ** 2)
        self.c = mu

    def get_symbolic_model(self):
        r"""Usage: """
        return self.symbolic_function

    def affine_update(self, other, grow_factor: float):
        if isinstance(other, EllipsoidalSet):
            Q_new = (1-grow_factor) * self.Q + grow_factor * other.Q
            c_new = (1-grow_factor) * self.c + grow_factor * other.c
            self.Q = Q_new
            self.c = c_new
        else:
            raise NotImplementedError

    def subtract(
            self,
            other
    ) -> None:
        r"""Minkowski difference of two ellipsoids."""
        if isinstance(other, EllipsoidalSet):
            c_new = self.c - other.c

            # clip diagonal elemets for numerical stability..
            # self_Q = np.where(self.Q < 1e4, self.Q, 0.0)
            # other_Q = np.where(other.Q < 1e4, other.Q, 0.0)

            p = np.sqrt(np.trace(self.Q) / np.trace(other.Q) + 1e-8)
            Q_tilde = (1 - 1. / p) * self.Q + (1 - p) * other.Q

            assert is_pos_def(Q_tilde), "result of Minkowski diff is not pos.def."

            self.c = c_new
            self.Q = Q_tilde
        else:
            raise NotImplementedError

    def transform(self,
                  A: np.ndarray,  # matrix,
                  **kwargs):
        self.Q = A @ self.Q @ A.T
        self.c = A @ self.c

    def set_center(self, c_new):
        assert c_new.reshape(-1).shape == self.c.reshape(-1).shape, \
            f"Shape mismatch. Expected {self.c.shape} but got {c_new.shape}"
        self.c = c_new.reshape(-1)

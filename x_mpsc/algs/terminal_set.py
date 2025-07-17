import time
from typing import Optional
import cvxpy as cp
import gymnasium as gym
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# local imports
import x_mpsc.common.loggers as loggers
from x_mpsc.common.sets import EllipsoidalSet, BoxSet


class DelayBuffer(object):
    def __init__(self, delay_factor: int):
        self.delay_factor = delay_factor
        self.ptr = 0
        self.length = 0
        self.As = []
        self.bs = []
        self.pts = []
        self.sims = []

    def __len__(self):
        return len(self.As)

    def add(self, A, b, points, simplices):
        if len(self) == 0:
            for _ in range(self.delay_factor):
                self.As.append(A)
                self.bs.append(b)
                self.pts.append(points)
                self.sims.append(simplices)
        else:
            self.As[self.ptr] = A
            self.bs[self.ptr] = b
            self.pts[self.ptr] = points
            self.sims[self.ptr] = simplices
        self.ptr = (self.ptr + 1) % self.delay_factor

    def get_delayed_data(self):
        return self.As[self.ptr], self.bs[self.ptr], \
               self.pts[self.ptr], self.sims[self.ptr]

    def get_most_recent_data(self):
        idx = (self.ptr - 1 + self.delay_factor) % self.delay_factor
        return self.As[idx], self.bs[idx], self.pts[idx], self.sims[idx]


class TerminalSet(object):
    def __init__(
            self,
            delay_factor: int,
    ):
        r"""Calculates the convex hull of given points x_i for i=1,..., N in
        form of polytopical constraints Ax<=b.
        """
        self.delay_factor = delay_factor
        self.A = None
        self.b = None
        self.buffer = DelayBuffer(delay_factor)

    def __repr__(self):
        return f"Ellipsoid with b={self.b}) and A=\n{self.A}"

    def calculate_convex_hull(self, pts):
        loggers.debug(f"Compute convex hull...")
        ts = time.time()
        # self.pts = pts
        data_dim = pts.shape[1]
        if data_dim > 1:
            hull = ConvexHull(pts)
            A = hull.equations[:, 0:data_dim]
            # Negative moves b to the RHS of the inequality:
            b = -hull.equations[:, data_dim]
        else:
            raise NotImplementedError
            # A = np.array([[1.], [-1]])
            # b1 = np.max(pts)
            # b2 = -np.min(pts)
            # b = np.array([b1, b2])
        loggers.debug(f"Done! (took: {(time.time()-ts):0.3f}s")
        return A, b, pts, hull.simplices

    @classmethod
    def can_reduce(cls, box: BoxSet) -> bool:
        box_can_be_reduced = not np.all(box.b < 1e5)
        return box_can_be_reduced

    # def compute_inscribed_ellipsoid(self) -> EllipsoidalSet:
    #     r"""Compute the Löwner-John inner ellipsoidal approx. of a polytope"""
    #     assert self.A is not None
    #     assert self.b is not None
    #     A = self.A
    #     b = self.b
    #     nc, nx = A.shape  # nc constraints, nx state dimension
    #     C_var = cp.Variable((nx, nx), PSD=True)  # EllipsoidalSet
    #     d_var = cp.Variable((nx, 1))  # Center
    #     constraints = [cp.norm(C_var @ A[i], 2) <= b[i] - A[i] @ d_var for i in
    #                    range(nc)]
    #     prob = cp.Problem(cp.Minimize(-cp.log_det(C_var)), constraints)
    #     prob.solve()
    #     C = C_var.value
    #     d = d_var.value.flatten()
    #     new_ellipsoid = EllipsoidalSet(Q=C @ C.T, c=d)
    #     return new_ellipsoid

    def draw_convex_hull(
            self,
            ax: plt.Axes,
            color: Optional[str] = 'blue',
            linestyle: Optional[str] = '--',
            dims: tuple = (0, 1),  # dimensions that are plotted
            **kwargs
    ):
        # if self.pts is None or self.simplices_most_recent is None:
        #     return
        A, b, pts, simplices = self.buffer.get_most_recent_data()
        for simplex in simplices:
            ax.plot(pts[simplex, 0], pts[simplex, 1],
                    color='grey', linestyle=linestyle, **kwargs)

        A, b, pts, simplices = self.buffer.get_delayed_data()
        for simplex in simplices:
            ax.plot(pts[simplex, 0], pts[simplex, 1],
                    color=color, linestyle=linestyle, **kwargs)

    @classmethod
    def extend_polytope(cls, A: np.ndarray, box: BoxSet):
        A_extended = np.zeros((A.shape[0], box.dim))
        d = cls.get_non_reducable_state_dims(box)
        A_extended[:, d] = A

        return A_extended

    @classmethod
    def get_non_reducable_state_dims(cls, box: BoxSet) -> np.ndarray:
        assert np.ndim(box.b) == 1, 'Expecting b to be vector for: Ax <= b.'
        non_reducable_bounds = box.b < 1e5
        non_reducable_state_dims = non_reducable_bounds[:non_reducable_bounds.size//2]
        return non_reducable_state_dims

    def solve(
            self,
            data: np.ndarray,
            space: gym.Space,
    ) -> None:
        r"""Calculate convex hull of given data. Then fit inner ellipsoid."""
        assert data.ndim == 2, f'Got wrong input data. Got: {data.shape}'
        state_box = BoxSet(from_space=space)
        if self.can_reduce(state_box):
            loggers.debug(f"Reduce state space box")
            non_reducable_state_dims = self.get_non_reducable_state_dims(state_box)
            xs = data[:, non_reducable_state_dims]
        else:
            xs = data

        A, b, pts, simplices = self.calculate_convex_hull(xs)

        if self.can_reduce(state_box):
            A = self.extend_polytope(A, state_box)

        self.buffer.add(A, b, pts, simplices)

        A, b, pts, simplices = self.buffer.get_delayed_data()
        self.A = A
        self.b = b







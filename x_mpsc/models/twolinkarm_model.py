from __future__ import annotations

from typing import Tuple

import numpy as np
import casadi as cs

# local imports
from matplotlib import pyplot as plt
from x_mpsc.common.utils import to_matrix
from x_mpsc.envs.twolinkarm import TwoLinkArmEnv


def get_C(a: cs.MX, thetas: cs.MX, theta_dots: cs.MX):
    """A matrix holding centrifugal and Coriolis forces. C has shape 2x1."""
    c = a[1] * cs.sin(thetas[1])
    c_1 = -c * theta_dots[1] * (2*theta_dots[0] + theta_dots[1])
    c_2 = c * theta_dots[0]**2
    return cs.vertcat(c_1, c_2)


def get_M(a: cs.MX, thetas: cs.MX) -> cs.MX:
    """The positive definite symmetric inertia matrix. M has shape 2x2."""
    cos_theta_2 = cs.cos(thetas[1])
    # m_11 = a[0] + 2 * a[1] * cos_theta_2
    # m_12 = a[2] + a[1] * cos_theta_2
    # m_21 = a[2] + a[1] * cos_theta_2
    # m_22 = a[2]
    m1 = cs.vertcat(a[0] + 2 * a[1] * cos_theta_2, a[2] + a[1] * cos_theta_2)
    m2 = cs.vertcat(a[2] + a[1] * cos_theta_2, a[2])
    return cs.horzcat(m1, m2)


class TwoLinkArmModel(object):

    def __init__(self, x_sym: cs.MX, u_sym: cs.MX, uncertainty: float = 0.8):
        self.env = TwoLinkArmEnv()
        self.nx = self.env.observation_space.shape[0]
        self.force_factor = self.env.force_factor
        self.dt = self.env.dt

        I = uncertainty * np.array(self.env.I)
        m = uncertainty * np.array(self.env.m)
        self.l = l = uncertainty * np.array(self.env.l)
        s = uncertainty * np.array(self.env.s)

        self.a = cs.MX(np.array(
            [I[0] + I[1] + m[1] * l[0] ** 2,
             m[1] * l[0] * s[1],
             I[1]]))

        self.B = np.array([[0.05, 0.025],
                           [0.025, 0.05]])

        self.x_sym = x_sym
        self.u_sym = u_sym

        next_state = self.get_casadi_function()
        self.f_discrete_func = cs.Function(
            'f_discrete_func',
            [self.x_sym, self.u_sym],
            [next_state, ]
        )

    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.predict_next_state(x, u)

    def predict_next_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        output = self.f_discrete_func(to_matrix(x), to_matrix(u))[:self.nx].full()
        return output

    def get_casadi_function(self):
        theta_1 = self.x_sym[4] * 10
        theta_2 = self.x_sym[5] * 10

        theta = cs.vertcat(theta_1, theta_2)
        theta_dot = self.x_sym[6:8] * 10

        goal_xy = self.x_sym[0:2] - self.x_sym[2:4]

        u = self.u_sym * self.force_factor

        M = get_M(self.a, theta)
        C = get_C(self.a, theta, theta_dot)

        # theta_acc = np.linalg.pinv(M) @ (u - C - self.B @ theta_dot)
        theta_acc = cs.solve(M, u - C - self.B @ theta_dot)

        new_thetas = theta + self.dt * theta_dot
        new_theta_dots = theta_dot + self.dt * theta_acc

        # reduced_state = cs.horzcat(new_thetas, new_theta_dots).T

        x2, y2 = self.get_end_effector_position(
            thetas=new_thetas)

        return cs.vertcat(
            x2,
            y2,
            x2 - goal_xy[0],
            y2 - goal_xy[1],
            new_thetas[0] / 10,
            new_thetas[1] / 10,
            new_theta_dots[0] / 10,
            new_theta_dots[1] / 10,
            0, 0, 0, 0,
            0, 0, 0, 0
        )

    def get_end_effector_position(
            self,
            thetas: cs.MX
    ) -> Tuple[cs.MX, cs.MX]:
        cos_theta_1 = cs.cos(thetas[0])
        sin_theta_1 = cs.sin(thetas[0])
        theta_sum = thetas[0] + thetas[1]
        x1 = cos_theta_1 * self.l[0]
        y1 = sin_theta_1 * self.l[0]
        x2 = x1 + cs.cos(theta_sum) * self.l[1]
        y2 = y1 + cs.sin(theta_sum) * self.l[1]
        return x2, y2


class Trajetories:
    def __init__(self):
        self.next_xs = []
        self.next_model_xs = []
        self.next_casadi_xs = []
        self.us = []

    def append(self, x_next_env, x_next_model, x_next_casadi, u):
        self.next_xs.append(x_next_env)
        self.next_model_xs.append(x_next_model)
        self.next_casadi_xs.append(x_next_casadi)
        self.us.append(u)

    def finalize(self):
        self.next_xs = np.array(self.next_xs)
        self.next_model_xs = np.array(self.next_model_xs)
        self.next_casadi_xs = np.array(self.next_casadi_xs)
        self.us = np.array(self.us)


if __name__ == '__main__':
    env = TwoLinkArmEnv()
    nx, nu = 8, 2
    x_sym = cs.MX.sym('x', nx, 1)
    u_sym = cs.MX.sym('u', nu, 1)
    model = TwoLinkArmModel(x_sym, u_sym, uncertainty=0.90)
    x, _ = env.reset()
    x_model = x
    x_casadi = x

    trajs = Trajetories()
    horizon = 100

    for i in range(horizon-1):
        u = env.action_space.sample()
        trajs.append(x, x_model, x_casadi, u)

        x, *_ = env.step(u)
        x_model = model.predict_next_state(x_model, u)
        x_model = x
        x_casadi = model.predict_next_state(x_casadi, u).flatten()
    trajs.append(x, x_model, x_casadi, u)

    trajs.finalize()
    fig, axes = plt.subplots(nrows=4, ncols=3)

    axes[0, 0].set_title("X Position [m]")
    axes[0, 0].plot(np.arange(horizon), trajs.next_xs[:, 0])
    axes[0, 0].plot(np.arange(horizon), trajs.next_model_xs[:, 0])
    axes[0, 0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 0])

    axes[1, 0].set_title("Y Position [m]")
    axes[1, 0].plot(np.arange(horizon), trajs.next_xs[:, 1])
    axes[1, 0].plot(np.arange(horizon), trajs.next_model_xs[:, 1])
    axes[1, 0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 1])

    axes[2, 0].set_title("X-Diff to target [m]")
    axes[2, 0].plot(np.arange(horizon), trajs.next_xs[:, 2])
    axes[2, 0].plot(np.arange(horizon), trajs.next_model_xs[:, 2])
    axes[2, 0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 2])

    axes[3, 0].set_title("Y-Diff to Target [m]")
    axes[3, 0].plot(np.arange(horizon), trajs.next_xs[:, 3])
    axes[3, 0].plot(np.arange(horizon), trajs.next_model_xs[:, 3])
    axes[3, 0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 3])

    axes[0, 1].set_title("Theta 1")
    axes[0, 1].plot(np.arange(horizon), trajs.next_xs[:, 4])
    axes[0, 1].plot(np.arange(horizon), trajs.next_model_xs[:, 4])
    axes[0, 1].plot(np.arange(horizon), trajs.next_casadi_xs[:, 4])

    axes[2, 1].set_title("Theta 2")
    axes[2, 1].plot(np.arange(horizon), trajs.next_xs[:, 5])
    axes[2, 1].plot(np.arange(horizon), trajs.next_model_xs[:, 5])
    axes[2, 1].plot(np.arange(horizon), trajs.next_casadi_xs[:, 5])

    axes[0, 2].set_title("Theta 1 Dot [deg/s]")
    axes[0, 2].plot(np.arange(horizon), trajs.next_xs[:, 6]*180/np.pi)
    axes[0, 2].plot(np.arange(horizon), trajs.next_model_xs[:, 6]*180/np.pi)
    axes[0, 2].plot(np.arange(horizon), trajs.next_casadi_xs[:, 6]*180/np.pi)

    axes[1, 2].set_title("Theta 2 Dot [deg/s]")
    axes[1, 2].plot(np.arange(horizon), trajs.next_xs[:, 7]*180/np.pi)
    axes[1, 2].plot(np.arange(horizon), trajs.next_model_xs[:, 7]*180/np.pi)
    axes[1, 2].plot(np.arange(horizon), trajs.next_casadi_xs[:, 7]*180/np.pi)

    # plot control inputs
    axes[3, 2].set_title("Action")
    axes[3, 2].plot(np.arange(trajs.us.shape[0]), trajs.us[:, 0])
    axes[3, 2].plot(np.arange(trajs.us.shape[0]), trajs.us[:, 1])

    plt.tight_layout()
    plt.show()

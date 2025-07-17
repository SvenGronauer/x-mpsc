from __future__ import annotations
import numpy as np
import casadi as cs

# local imports
from matplotlib import pyplot as plt
from x_mpsc.common.utils import to_matrix
from x_mpsc.envs.cartpole import CartPoleEnv


class CartPoleModel(object):

    def __init__(self, x_sym: cs.MX, u_sym: cs.MX, uncertainty: float = 0.8):
        self.env = CartPoleEnv(noise=False)
        self.nx = self.env.observation_space.shape[0]
        self.max_speed = self.env.max_speed
        self.max_torque = self.env.max_torque
        self.dt = self.env.dt
        self.g = self.env.g
        self.m = uncertainty * self.env.m
        self.length = uncertainty * self.env.length
        self.masscart = uncertainty * self.env.masscart
        self.masspole = uncertainty * self.env.masspole

        self.x_sym = x_sym
        self.u_sym = u_sym

        next_state = self.get_casadi_function()
        self.f_discrete_func = cs.Function(
            'f_discrete_func',
            [self.x_sym, self.u_sym],
            [next_state, ]
        )

    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        assert x.ndim == u.ndim
        if x.ndim == 1:
            x, x_dot, theta, theta_dot = x
        elif x.ndim == 2:
            x, x_dot, theta, theta_dot = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        else:
            raise ValueError

        gravity = self.g

        masstotal = (self.masspole + self.masscart)
        length = self.length
        force_mag = 10.0
        mpl = self.masspole * length
        friction = 0.1
        tau = self.dt

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        u = np.clip(u, -1, 1) * force_mag

        xdot_update = (-2 * mpl * (
                theta_dot ** 2) * sin_theta + 3 * self.masspole * gravity * sin_theta * cos_theta + 4 * u - 4 * friction * x_dot) / (
                              4 * masstotal - 3 * self.masspole * cos_theta ** 2)
        thetadot_update = (-3 * mpl * (
                theta_dot ** 2) * sin_theta * cos_theta + 6 * masstotal * gravity * sin_theta + 6 * (
                                   u - friction * x_dot) * cos_theta) / (
                                  4 * length * masstotal - 3 * mpl * cos_theta ** 2)

        x_new = x + x_dot * tau
        theta_new = theta + theta_dot * tau
        x_dot_new = x_dot + xdot_update * tau
        theta_dot_new = theta_dot + thetadot_update * tau

        if x.ndim == 1:
            next_state = np.array([x_new, x_dot_new, theta_new, theta_dot_new],
                              dtype=np.float32).flatten()
        else:
            next_state = np.array([x_new, x_dot_new, theta_new, theta_dot_new],
                              dtype=np.float32).T

        return next_state

    def predict_next_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        output = self.f_discrete_func(to_matrix(x), to_matrix(u))[:self.nx].full()
        return output

    def get_casadi_function(self):
        x = self.x_sym[0]
        x_dot = self.x_sym[1]
        theta = self.x_sym[2]
        theta_dot = self.x_sym[3]

        gravity = self.g
        length = self.length
        masstotal = (self.masspole + self.masscart)
        force_mag = 10.0
        mpl = self.masspole * self.length
        friction = 0.1
        tau = self.dt

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        u = self.u_sym[0] * force_mag

        xdot_update = (-2 * mpl * (
                theta_dot ** 2) * sin_theta + 3 * self.masspole * gravity * sin_theta * cos_theta + 4 * u - 4 * friction * x_dot) / (
                              4 * masstotal - 3 * self.masspole * cos_theta ** 2)
        thetadot_update = (-3 * mpl * (
                theta_dot ** 2) * sin_theta * cos_theta + 6 * masstotal * gravity * sin_theta + 6 * (
                                   u - friction * x_dot) * cos_theta) / (
                                  4 * length * masstotal - 3 * mpl * cos_theta ** 2)

        x_new = x + x_dot * tau
        theta_new = theta + theta_dot * tau
        x_dot_new = x_dot + xdot_update * tau
        theta_dot_new = theta_dot + thetadot_update * tau

        next_state = cs.vertcat(x_new, x_dot_new, theta_new, theta_dot_new,
                                0, 0, 0, 0)
        return next_state


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
    env = CartPoleEnv(noise=False)
    nx, nu = 4, 1
    x_sym = cs.MX.sym('x', nx, 1)
    u_sym = cs.MX.sym('u', nu, 1)
    model = CartPoleModel(x_sym, u_sym, uncertainty=0.8)
    x = env.reset(state=np.zeros(nx))
    x_model = x
    x_casadi = x
    u = np.ones(nu)

    trajs = Trajetories()

    horizon = 100
    for i in range(horizon-1):
        u = env.action_space.sample()
        trajs.append(x, x_model, x_casadi, u)

        x, *_ = env.step(u)
        x_model = model.forward(x_model, u)
        x_casadi = model.predict_next_state(x_casadi, u).flatten()
    trajs.append(x, x_model, x_casadi, u)

    trajs.finalize()
    fig, axes = plt.subplots(nrows=3)
    axes[0].set_title("Angle [deg]")
    axes[0].plot(np.arange(horizon), trajs.next_xs[:, 0]*180/np.pi)
    axes[0].plot(np.arange(horizon), trajs.next_model_xs[:, 0]*180/np.pi)
    axes[0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 0]*180/np.pi)
    axes[1].set_title("Speed [deg/s]")
    axes[1].plot(np.arange(horizon), trajs.next_xs[:, 1]*180/np.pi)
    axes[1].plot(np.arange(horizon), trajs.next_model_xs[:, 1]*180/np.pi)
    axes[1].plot(np.arange(horizon), trajs.next_casadi_xs[:, 1]*180/np.pi)

    # plot control inputs
    axes[2 ].set_title("Action")
    axes[2].plot(np.arange(trajs.us.size), trajs.us.flatten())

    plt.tight_layout()
    plt.show()

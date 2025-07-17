r"""The Simple Pendulum from Underactuated Robotics (Russ Tedrake).

http://underactuated.mit.edu/pend.html
"""
from __future__ import annotations
from os import path
from typing import Optional
import numpy as np
import casadi as cs

# local imports
from matplotlib import pyplot as plt
from x_mpsc.common.utils import to_matrix
from x_mpsc.envs.simple_pendulum.pendulum import SimplePendulumEnv


class SimplePendulumModel(object):

    def __init__(self, x_sym: cs.MX, u_sym: cs.MX, uncertainty: float = 0.8):
        self.env = SimplePendulumEnv(noise=False)
        self.nx = self.env.observation_space.shape[0]
        self.max_speed = self.env.max_speed
        self.max_torque = self.env.max_torque
        self.dt = self.env.dt
        self.g = self.env.g
        self.m = uncertainty * self.env.m
        self.l = uncertainty * self.env.l
        self.damping = uncertainty * self.env.damping

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
            th, thdot = x
        elif x.ndim == 2:
            th, thdot = x[:, 0], x[:, 1]
        else:
            raise ValueError

        torque = np.clip(u, -self.max_torque, self.max_torque)

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        b = self.damping
        acc = (torque + m * g * l * np.sin(th) - b * thdot) / (m * l ** 2)
        newthdot = thdot + acc * dt
        newth = th + newthdot * dt
        if x.ndim == 1:
            next_state = np.array([newth, newthdot], dtype=np.float32).flatten()
        else:
            next_state = np.array([newth, newthdot], dtype=np.float32).T

        return next_state

    def predict_next_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        output = self.f_discrete_func(to_matrix(x), to_matrix(u))[:self.nx].full()
        return output

    def get_casadi_function(self):
        th = self.x_sym[0]
        thdot = self.x_sym[1]
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        b = self.damping
        acc = (self.u_sym + m * g * l * np.sin(th) - b * thdot) / (m * l ** 2)
        newthdot = thdot + acc * dt
        newth = th + newthdot * dt
        next_state = cs.vertcat(newth, newthdot, 0, 0)

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
    env = SimplePendulumEnv(noise=False)
    nx, nu = 2, 1
    x_sym = cs.MX.sym('x', nx, 1)
    u_sym = cs.MX.sym('u', nu, 1)
    model = SimplePendulumModel(x_sym, u_sym, uncertainty=2.5)
    x = env.reset(state=np.zeros(2))
    u = np.ones(1)

    trajs = Trajetories()
    u = env.action_space.sample()
    trajs.append(x, x, x, u)

    horizon = 100
    for i in range(horizon-1):
        x_next_env, *_ = env.step(u)
        x_next_model = model.forward(x, u)
        x_next_casadi = model.predict_next_state(x, u).flatten()
        x = x_next_env
        u = env.action_space.sample()
        trajs.append(x_next_env, x_next_model, x_next_casadi, u)

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

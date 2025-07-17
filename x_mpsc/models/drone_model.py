from __future__ import annotations
import numpy as np
import casadi as cs

# local imports
from matplotlib import pyplot as plt
from x_mpsc.common.utils import to_matrix
from x_mpsc.envs.drone import DroneEnv


def rad2deg(x):
    return x * 180 / np.pi


class DroneModel(object):

    def __init__(self, x_sym: cs.MX, u_sym: cs.MX, uncertainty: float = 0.8):
        self.env = DroneEnv(noise=False)
        self.nx = self.env.observation_space.shape[0]

        self.dt = self.env.dt
        self.g = self.env.g
        self.m = uncertainty * self.env.m

        self.force_torque_factor = uncertainty * self.env.drone.force_torque_factor
        self.thrust_factor = uncertainty * self.env.drone.thrust_factor

        self.I = uncertainty * np.diag([1.33e-5, 1.33e-5, 2.64e-5])

        self.x_sym = x_sym
        self.u_sym = u_sym

        next_state = self.get_casadi_function()
        self.f_discrete_func = cs.Function(
            'f_discrete_func',
            [self.x_sym, self.u_sym],
            [next_state, ]
        )

    def forward(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.predict_next_state(x, u).flatten()

    def predict_next_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        output = self.f_discrete_func(to_matrix(x), to_matrix(u))[:self.nx].full()
        return output

    def get_casadi_function(self):
        xyz = self.x_sym[0:3]
        rpy = self.x_sym[3:6]
        xyz_dot = self.x_sym[6:9]
        rpy_dot = self.x_sym[9:12]

        collective_thrust = 4 * 0.075 * (self.u_sym[0] + 1)
        gravity = cs.MX(np.array([[0], [0], [self.g]])) *self.m

        sr = cs.sin(self.x_sym[3])
        sp = cs.sin(self.x_sym[4])
        sy = cs.sin(self.x_sym[5])
        cr = cs.cos(self.x_sym[3])
        cp = cs.cos(self.x_sym[4])
        cy = cs.cos(self.x_sym[5])
        R_times_z = cs.vertcat(sy*sr+cy*cr*sp, -cr*sr+sy*sp*cr, cp*cr)
        
        xyz_new = xyz + xyz_dot * self.dt
        xyz_dot_new = xyz_dot + (R_times_z * collective_thrust - gravity)/self.m * self.dt
        rpy_new = (rpy + rpy_dot * self.dt)
        rpy_dot_new = rpy_dot + self.u_sym[1:4] * np.pi/3 * self.dt

        next_state = cs.vertcat(xyz_new, rpy_new, xyz_dot_new, rpy_dot_new,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
    env = DroneEnv(noise=False)
    nx, nu = 12, 4
    x_sym = cs.MX.sym('x', nx, 1)
    u_sym = cs.MX.sym('u', nu, 1)
    model = DroneModel(x_sym, u_sym, uncertainty=0.999)
    x, _ = env.reset()
    x_model = x
    x_casadi = x
    u = np.ones(nu)

    trajs = Trajetories()

    horizon = 250
    for i in range(horizon-1):
        # print(x)
        u = env.safe_controller(x)
        # u = np.array([0.01, 0, 0, 0])
        trajs.append(x, x_model, x_casadi, u)

        x, *_ = env.step(u)
        x_model = model.forward(x_model, u)
        x_casadi = model.predict_next_state(x_casadi, u).flatten()
    trajs.append(x, x_model, x_casadi, u)

    trajs.finalize()
    fig, axes = plt.subplots(nrows=3, ncols=4)

    axes[0, 0].set_title("X Position [m]")
    axes[0, 0].plot(np.arange(horizon), trajs.next_xs[:, 0])
    axes[0, 0].plot(np.arange(horizon), trajs.next_model_xs[:, 0])
    axes[0, 0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 0])
    axes[1, 0].set_title("Y Position [m]")
    axes[1, 0].plot(np.arange(horizon), trajs.next_xs[:, 1])
    axes[1, 0].plot(np.arange(horizon), trajs.next_model_xs[:, 1])
    axes[1, 0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 1])
    axes[2, 0].set_title("Z [m]")
    axes[2, 0].plot(np.arange(horizon), trajs.next_xs[:, 2])
    axes[2, 0].plot(np.arange(horizon), trajs.next_model_xs[:, 2])
    axes[2, 0].plot(np.arange(horizon), trajs.next_casadi_xs[:, 2])

    axes[0, 1].set_title("X Speed [m/s]")
    axes[0, 1].plot(np.arange(horizon), trajs.next_xs[:, 6])
    axes[0, 1].plot(np.arange(horizon), trajs.next_model_xs[:, 6])
    axes[0, 1].plot(np.arange(horizon), trajs.next_casadi_xs[:, 6])
    axes[1, 1].set_title("Y Speed [m/s]")
    axes[1, 1].plot(np.arange(horizon), trajs.next_xs[:, 7])
    axes[1, 1].plot(np.arange(horizon), trajs.next_model_xs[:, 7])
    axes[1, 1].plot(np.arange(horizon), trajs.next_casadi_xs[:, 7])
    axes[2, 1].set_title("Z Speed [m/s]")
    axes[2, 1].plot(np.arange(horizon), trajs.next_xs[:, 8])
    axes[2, 1].plot(np.arange(horizon), trajs.next_model_xs[:, 8])
    axes[2, 1].plot(np.arange(horizon), trajs.next_casadi_xs[:, 8])

    axes[0, 2].set_title("Roll [deg]")
    axes[0, 2].plot(np.arange(horizon), rad2deg(trajs.next_xs[:, 3]))
    axes[0, 2].plot(np.arange(horizon), rad2deg(trajs.next_model_xs[:, 3]))
    axes[0, 2].plot(np.arange(horizon), rad2deg(trajs.next_casadi_xs[:, 3]))
    axes[1, 2].set_title("Pitch [deg]")
    axes[1, 2].plot(np.arange(horizon), rad2deg(trajs.next_xs[:, 4]))
    axes[1, 2].plot(np.arange(horizon), rad2deg(trajs.next_model_xs[:, 4]))
    axes[1, 2].plot(np.arange(horizon), rad2deg(trajs.next_casadi_xs[:, 4]))
    axes[2, 2].set_title("Yaw [deg]")
    axes[2, 2].plot(np.arange(horizon), rad2deg(trajs.next_xs[:, 5]))
    axes[2, 2].plot(np.arange(horizon), rad2deg(trajs.next_model_xs[:, 5]))
    axes[2, 2].plot(np.arange(horizon), rad2deg(trajs.next_casadi_xs[:, 5]))

    axes[0, 3].set_title("Roll Rate [rad/s]")
    axes[0, 3].plot(np.arange(horizon), trajs.next_xs[:, 6])
    axes[0, 3].plot(np.arange(horizon), trajs.next_model_xs[:, 6])
    axes[0, 3].plot(np.arange(horizon), trajs.next_casadi_xs[:, 6])
    axes[1, 3].set_title("Pitch Rate [rad/s]")
    axes[1, 3].plot(np.arange(horizon), trajs.next_xs[:, 7])
    axes[1, 3].plot(np.arange(horizon), trajs.next_model_xs[:, 7])
    axes[1, 3].plot(np.arange(horizon), trajs.next_casadi_xs[:, 7])
    axes[2, 3].set_title("Yaw Rate [rad/s]")
    axes[2, 3].plot(np.arange(horizon), trajs.next_xs[:, 8])
    axes[2, 3].plot(np.arange(horizon), trajs.next_model_xs[:, 8])
    axes[2, 3].plot(np.arange(horizon), trajs.next_casadi_xs[:, 8])

    # plot control inputs
    # axes[2, 3].set_title("Action")
    # axes[2, 3].plot(np.arange(trajs.us.shape[0]), trajs.us[:, 0])
    # axes[2, 3].plot(np.arange(trajs.us.shape[0]), trajs.us[:, 1])

    plt.tight_layout()
    plt.show()
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# local imports
from x_mpsc.envs.linear import base


class MassSpringDamperEnv(base.LinearQuadraticSystemEnv):
    """A system with linear dynamics and quadratic costs.

    According to the specifications given in:
        Kim Wabersich and Melanie Zeilinger
        Linear model predictive control safety certification for learning-based
        control
    """
    def __init__(self,
                 x0=None,
                 dt: float = 0.1  # [s]
                 ):
        spring_damp = -3
        self.dt = dt
        a = dt * spring_damp
        A = np.array([[1, dt],
                      [a, 0.8]], dtype=np.float64)
        B = np.array([[0],
                      [dt]], dtype=np.float64)
        Q = np.eye(A.shape[0])
        R = np.eye(B.shape[1])

        pos_lim = 1.0
        vel_lim = 0.5

        super().__init__(
            A, B, Q, R,
            act_dim=1,
            x0=x0,
            noise=-1,  # disables the noise by default
            u_max=2.0,
            observation_space_high=np.array([pos_lim, vel_lim]),  # pos, vel
            observation_space_low=np.array([-pos_lim, -vel_lim]),  # pos, vel
        )

    def nominal(self):
        """Changes system parameters to the nominal model."""
        self.A = np.array([[1.0, 0.1], [-0.23, 0.78]], dtype=np.float64)
        self.noise = -1  # negative value means zero noise
        return self

    def reset(
            self,
            x: Optional[np.ndarray]=None
    ):
        if x is None:
            # todo @ Sven: these boundaries might be adjusted
            pos = np.array([-.8, .8])
            x1 = np.random.uniform(*pos)
            vel = np.array([-0.4, 0.4])
            x2 = np.random.uniform(*vel)
            x = np.array([x1, x2])
        return super().reset(x=x)


class MassSpringDamperTracker(gym.Wrapper):
    r"""Track state and input trajectories by wrapping the env.

    Code Usage:
        >>> env = MassSpringDamperEnv()
        >>> env = MassSpringDamperTracker(env)
    """

    def __init__(
            self,
            env: MassSpringDamperEnv,
    ):
        super().__init__(env)
        self.ax = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def _after_step(self, x, r, terminated, truncated, info):
        self.states.append(x)
        self.rewards.append(r)
        self.dones.append(terminated)

    def _before_step(self, action):
        self.actions.append(action)

    @classmethod
    def _plot_states(cls, ax, xs, cmap='twilight', color='blue', marker='.'):
        states = np.array(xs)
        if len(states) > 1:
            n = states.shape[0]
            x1 = states[:, 0]
            x2 = states[:, 1]
            ax.plot(x1, x2, color=color, marker=marker)
            # plt.scatter(x1, x2, c=t, cmap=cmap, marker='.',)

    def _plot_constraints(self, ax, **kwargs):
        # draw state space constraints
        lows = self.env.observation_space.low
        highs = self.env.observation_space.high
        # lim_pos = constraints[0]
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.25, 1.25)
        x = ([lows[0], highs[0], highs[0], lows[0], lows[0]])
        y = ([highs[1], highs[1], lows[1], lows[1], highs[1]])
        ax.plot(x, y, **kwargs)
        ax.set_xlabel('Position [m]')
        ax.set_ylabel('Velocity [m/s]')

    def plot_episode(self, color='blue', marker='.', nominal_states=None):
        """Display trajectories in GUI."""
        # if states_nominal is not None:
        #     # over-write tracked nomimal states:
        #     self.states_nominal = np.array(states_nominal)
        if self.ax is None:
            fig = plt.figure()
            self.ax = fig.add_subplot(1, 1, 1)
        self._plot_constraints(self.ax, color='red', linestyle='dashed')

        self._plot_states(self.ax, self.states, color=color, marker=marker)
        # self._plot_states(ax, self.states_nominal, color='green')

        if nominal_states is not None:
            print(f'Got nominal states with shape :{nominal_states.shape}')
            self._plot_states(self.ax, nominal_states.T, color='cyan')
            plt.show()
            self.ax = None

        # self._plot_constraints(ax, self.state_constraints_real, color='blue')
        plt.show()
        self.ax = None
        self.reset()

    def reset(self, **kwargs):
        self.actions = []
        self.states = []
        x = super().reset(**kwargs)
        self.states.append(x)
        return x, {}

    def step(self, action, u_nominal=None):
        self._before_step(action)
        x, r, terminated, truncated, info = super().step(action)
        self._after_step(x, r, terminated, truncated, info)
        return x, r, terminated, truncated, info

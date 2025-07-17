from typing import Optional, Union
import numpy as np
import gymnasium as gym


class LinearQuadraticSystemEnv(gym.Env):
    """Implements a system with linear dynamics and quadratic costs."""
    def __init__(
            self,
            A: np.ndarray,
            B: np.ndarray,
            Q: np.ndarray,
            R: np.ndarray,
            act_dim: int,
            u_max: float,
            x0=None,  # deterministic reset state
            noise=0.01,
            observation_space_high: Optional[np.ndarray] = None,
            observation_space_low: Optional[np.ndarray] = None
    ):
        assert A.ndim == 2, f'Expecting a matrix.'
        super(LinearQuadraticSystemEnv, self).__init__()
        self.act_dim = act_dim
        self.state_dim = A.shape[0]
        # state transition matrices for linear system:
        #     x(t+1) = A x(t) + B u(t)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        # Upper and lower limit for observation space
        assert observation_space_high.shape == observation_space_low.shape
        if observation_space_high is None:
            self.o_high = 1000. * np.ones((A.shape[0],), dtype=np.float64)
        else:
            self.o_high = observation_space_high.astype(np.float64)
        if observation_space_low is None:
            self.o_low = -1. * self.o_high
        else:
            self.o_low = observation_space_low.astype(np.float64)

        # Upper and lower limit for action space
        self.a_high = u_max * np.ones((act_dim,), dtype=np.float64)
        self.a_low = -self.a_high
        self.u_max = u_max

        # Create spaces
        self.obs_dim = self.A.shape[0]
        self.observation_space = gym.spaces.Box(self.o_low, self.o_high,
                                                dtype=np.float64)
        self.action_space = gym.spaces.Box(self.a_low, self.a_high,
                                           dtype=np.float64)


        # initial condition for system:
        #   if x0 is None, then randomly sample from observation space at each reset
        self.x0 = x0
        self.x = self.x0
        self.iteration = 0
        self.noise = noise

    def get_system_matrices(self):
        return self.A, self.B, self.Q, self.R

    def step(
            self,
            action: np.ndarray
    ):
        """Calculates one step of forward dynamics.

        State transition for linear systems:
            x(k+1) = A x(k) + B u(k) [+ w(k)]
        """
        self.iteration += 1
        assert self.x.ndim == 1, 'state must be a vector '
        # assert np.linalg.norm(action, np.inf) <= 1.0
        cost = float(self.cost_function(x=self.x, u=action))
        u = np.clip(action, -self.u_max, self.u_max)
        reward = -1. * cost
        next_x = self.A @ self.x + self.B @ u
        if self.noise > 0:
            next_x += np.random.uniform(-self.noise, self.noise, self.obs_dim)
        self.x = next_x
        terminated = False
        truncated = False
        info = dict()
        return np.squeeze(next_x), reward, terminated, truncated, info

    def cost_function(
            self,
            x: np.ndarray,
            u: Union[np.ndarray, float]
    ):
        if np.isscalar(u):
            u = np.array([u])
        c = x.T @ self.Q @ x + u.T @ self.R @ u
        return c

    def reset(
            self,
            x: Optional[np.ndarray]=None
    ):
        """ child classes can determine reset state."""
        self.iteration = 0
        if x is not None:
            self.x = x
        elif self.x0 is not None:  # use deterministic x0 given at __init__()
            self.x = self.x0
        else:
            self.x = self.observation_space.sample()
        return self.x, {}

    def render(self, mode='human'):
        pass





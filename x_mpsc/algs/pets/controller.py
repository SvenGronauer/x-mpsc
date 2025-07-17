r"""Models for Probabilistic Ensembles With trajectory sampling. (PETS).

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    28.04.2022
"""
import abc

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

# local imports
from x_mpsc.common import loggers
import x_mpsc.algs.core as core
from x_mpsc.models.ensemble import DynamicsModel
from x_mpsc.algs.pets.optimizer import CEMOptimizer


class Controller(nn.Module, abc.ABC):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int
    ):
        super(Controller, self).__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    def reset(
            self,
            obs: th.Tensor
    ) -> None:
        pass

    def step(self, obs: np.ndarray) -> np.ndarray:
        r"""Return an action."""
        raise NotImplementedError


class RandomController(Controller):
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            action_space: gym.Space,
            frame_skip: int = 1
    ):
        super(RandomController, self).__init__(
            obs_dim=obs_dim,
            act_dim=act_dim
        )
        # Note: additionally provide Action Space for sampling within boundaries
        self.action_space = action_space
        self.frame_skip = frame_skip
        self.num_frame = 0
        self.current_action = self.action_space.sample()

    def reset(self, obs: th.Tensor):
        self.num_frame = 0

    def step(self, obs: np.ndarray) -> np.ndarray:
        r"""Return an action."""
        if self.num_frame % self.frame_skip == 0:
            self.current_action = self.action_space.sample()
        self.num_frame += 1
        return self.current_action


class CrossEntropyMethodController(Controller):
    def __init__(
            self,
            model: DynamicsModel,
            env: gym.Env,
            alpha: float,
            trajectory_length: int,
            max_iters: int = 5,
            pop_size: int = 20,
            num_elites: int = 5,
    ):
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
        super(CrossEntropyMethodController, self).__init__(
            obs_dim=obs_dim,
            act_dim=act_dim
        )
        self.model = model  # get the Model Ensemble here
        self.trajectory_length = trajectory_length  # T
        self.action_sequence = np.zeros((trajectory_length, self.act_dim))
        assert hasattr(env, "calculate_reward"), \
            'Missing: calculate_reward() method in env'

        ac_ub, ac_lb = env.action_space.high, env.action_space.low
        self.init_var = np.square(ac_ub - ac_lb) / 16
        self.optimizer = CEMOptimizer(
            act_dim=act_dim,
            alpha=alpha,
            max_iters=max_iters,
            trajectory_length=trajectory_length,
            pop_size=pop_size,
            num_elites=num_elites,
            trajectory_reward_function=env.calculate_reward,
            upper_bound=env.action_space.high,
            lower_bound=env.action_space.low
        )

    def reset(
            self,
            obs: th.Tensor,
            use_multi_core: bool = False,
    ):
        r"""Reset action sequence and start warm-up."""
        # loggers.info(f"Reset CEM controller.")
        self.action_sequence = np.zeros((self.trajectory_length, self.act_dim))
        # do the warm up
        self.action_sequence = self.optimizer.obtain_solution(
            obs, self.action_sequence, self.init_var, self.model, use_multi_core
        )

    def step(
            self,
            obs: th.Tensor,
            run_on_multi_cores: bool = False,
    ) -> np.ndarray:
        r"""Return an action."""
        new_action_seq = self.optimizer.obtain_solution(
            obs, self.action_sequence, self.init_var, self.model,
            run_on_multi_cores=run_on_multi_cores
        )
        self.action_sequence[:-1] = new_action_seq[1:]
        return new_action_seq[0]  # return only first action of (a_0, ..., a_N)

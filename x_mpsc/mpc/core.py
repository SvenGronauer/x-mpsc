r"""Core classes and functions for safe reinforcement learning.

"""
import abc
import gymnasium as gym
import numpy as np
from typing import Tuple, Optional


class Solver(abc.ABC):
    r"""Solver that certifies action proposals."""
    def __init__(self,
                 env: gym.Env,
                 model,  #: SymbolicModel,
                 warm_start: bool = True,
                 debug: bool = False
                 ):
        self.debug = debug
        self.env = env
        self.model = model
        self.opti_dict = {}
        self.warm_start = warm_start

    @abc.abstractmethod
    def certify_action(self,
                       obs: np.ndarray,
                       uncertified_input: np.ndarray,  # u_L
                       target_state: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, bool]:
        r"""Check if system stays safe if proposed action is applied. Returns
        a safe action otherwise."""
        raise NotImplementedError

    @abc.abstractmethod
    def setup_optimizer(self):
        """Setup the underlying optimization problem."""
        raise NotImplementedError

    def solve(self,
              obs: np.ndarray,
              uncertified_input: np.ndarray,
              target_state: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, bool]:
        r"""Alias method for certify action."""
        return self.certify_action(obs, uncertified_input, target_state)


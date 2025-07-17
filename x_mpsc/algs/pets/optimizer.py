r"""Copied from:
https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/optimizers.py
"""
from typing import Callable

import torch as th

import numpy as np
import scipy.stats as stats

# local imports
import x_mpsc.common.mpi_tools as mpi
from x_mpsc.models.ensemble import DynamicsModel


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEMOptimizer(Optimizer):

    def __init__(
            self,
            act_dim: int,
            max_iters: int,
            trajectory_length: int,
            pop_size: int,
            num_elites,
            trajectory_reward_function: Callable,
            upper_bound=None,
            lower_bound=None,
            epsilon=0.001,
            alpha=0.25
    ):
        """Creates an instance of this class.

        Arguments:
            act_dim (int): The dimensionality of the action space
            max_iters (int): Number of iterations for optimization
            trajectory_length (int): Number of actions in trajectory
            pop_size (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.act_dim = act_dim
        self.trajectory_length = trajectory_length
        self.pop_size = pop_size
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.local_pop_size = pop_size // mpi.num_procs()
        self.local_num_elites = num_elites // mpi.num_procs()

        self.ub = upper_bound
        self.lb = lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.trajectory_reward_function = trajectory_reward_function

        if num_elites > pop_size:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(
            self,
            obs: th.Tensor,
            init_mean: np.ndarray,
            init_var: np.ndarray,
            model: DynamicsModel,
            run_on_multi_cores: bool = False,
    ) -> np.ndarray:
        """Optimizes the reward function using the provided initial candidate distribution

        Arguments:
            obs
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
            model: Model Ensemble
            iterations: int
        """
        N = self.trajectory_length
        if run_on_multi_cores:
            mean, var, t = mpi.mpi_avg(init_mean), mpi.mpi_avg(init_var), 0
        else:
            mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        pop_size = self.local_pop_size if run_on_multi_cores else self.pop_size
        num_elites = self.local_num_elites if run_on_multi_cores else self.num_elites

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            action_sequences = X.rvs(size=[pop_size, N, self.act_dim]) * np.sqrt(constrained_var) + mean
            action_sequences = action_sequences.astype(np.float32)

            trajectory_rewards = np.zeros(pop_size)
            for i, act_seq in enumerate(action_sequences):
                trajectory = model.sample_trajectory(obs, act_seq)
                rew = self.trajectory_reward_function(trajectory, act_seq)
                trajectory_rewards[i] = rew

            # argsort returns: min_value, ...., max_value
            best_trajectories = np.argsort(trajectory_rewards)
            elites = action_sequences[best_trajectories][-num_elites:]

            if run_on_multi_cores:
                new_mean = mpi.mpi_avg(np.mean(elites, axis=0))
                new_var = mpi.mpi_avg(np.var(elites, axis=0))
            else:
                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1

        return mean

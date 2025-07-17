""" PyTorch implementation of Proximal Policy Optimization (PPO) Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    10.10.2020
Updated:    15.11.2020
"""
import copy

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from copy import deepcopy
import warnings

import x_mpsc.algs.ppo.ppo as ppo
import x_mpsc.algs.utils as utils
import x_mpsc.algs.core as alg_core
import x_mpsc.common.mpi_tools as mpi_tools
import x_mpsc.algs.utils as U
import x_mpsc.common.loggers as loggers


class SafetyLayer:
    """Layer to learn constraint models and to impose action projection.
    """

    def __init__(self,
                 obs_space,
                 act_space,
                 hidden_dim=64,
                 num_constraints=1,
                 lr=0.001,
                 slack=None,
                 device='cpu',
                 **kwargs):
        # Parameters.
        self.num_constraints = num_constraints
        self.device = device
        # Seperate model per constraint.
        input_dim = obs_space.shape[0]
        output_dim = act_space.shape[0]

        # default 1 layer
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        elif isinstance(hidden_dim, list):
            hidden_dims = hidden_dim
        else:
            raise ValueError("hidden_dim can only be int or list.")
        self.constraint_model =  alg_core.build_mlp_network(
                sizes=[input_dim, hidden_dim, output_dim],
                activation="relu")
        #])
        # Constraint slack variables/values.
        assert slack is not None and isinstance(slack, (int, float, list))
        # if isinstance(slack, (int, float)):
        #     slack = [slack] * obs_space.shape[0]
        self.slack = np.array(slack)
        # Optimizers.
        self.optimizer =th.optim.Adam(self.constraint_model.parameters(), lr=lr)

    def to(self,
           device
           ):
        """Puts agent to device.
        """
        self.constraint_model.to(device)

    def train(self):
        """Sets training mode.
        """
        self.constraint_model.train()

    def eval(self):
        """Sets evaluation mode.
        """
        self.constraint_model.eval()

    def state_dict(self):
        """Snapshots agent state.
        """
        return {
            "constraint_models": self.constraint_model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self,
                        state_dict
                        ):
        """Restores agent state.
        """
        self.constraint_model.load_state_dict(state_dict["constraint_models"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def compute_loss(self,
                     batch
                     ):
        """Gets constraint value L2 loss for each constraint.
        """
        obs, act = batch["obs"].to(self.device), batch["act"].to(self.device)
        c, c_next = batch["c"].to(self.device), batch["c_next"].to(self.device)

        # gs = [model(obs) for model in self.constraint_models]

        g = self.constraint_model(obs)
        c_next_pred = th.sum(g * act, dim=1)

        # Each is (N,1,A) x (N,A,1) -> (N,), so [(N,)]_{n_constriants}
        # c_next_pred = [
        #     c[:, i] + th.bmm(g.view(g.shape[0], 1, -1),
        #                         act.view(act.shape[0], -1, 1)).view(-1)
        #     for i, g in enumerate(gs)
        # ]
        loss = th.mean(th.square(c_next - c_next_pred))
        # losses = [
        #     th.mean((c_next[:, i] - c_next_pred[i]) ** 2).cpu()
        #     for i in range(self.num_constraints)
        # ]
        return loss

    def update(self, batch):
        """Updates the constraint models from data batch.
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        results = {
            "constraint_loss": loss.item()
        }
        return results

    @th.no_grad()
    def get_safe_action(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            c: np.ndarray,
    ):
        """Does action projection with the trained safety layer.
        According to Dalal 2018, this simple projection works when only 1 constraint at a time
        is active; for multiple active constriants, either resort to in-graph QP solver such as
        OptLayer or see cvxpylayers (https://github.com/cvxgrp/cvxpylayers).
        Args:
            obs (th.FloatTensor): observations, shape (B,O).
            act (th.FloatTensor): actions, shape (B,A).
            c (th.FloatTensor): constraints, shape (B,C).

        Returns:
            th.FloatTensor: transformed/projected actions, shape (B,A).
        """
        self.eval()
        # [(B,A)]_C
        obs = th.as_tensor(obs, dtype=th.float32)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        act = th.as_tensor(act, dtype=th.float32)
        if len(act.shape) == 1:
            act = act.unsqueeze(0)
        c = th.as_tensor(c, dtype=th.float32)
        if len(c.shape) == 1:
            c = c.unsqueeze(0)
        g = self.constraint_model(obs)
        # Find the lagrange multipliers [(B,)]_C
        multipliers = []
        for i in range(len(g)):
            #g_i = g[i]  # (B,A)
            #c_i = c[:, i]  # (B,)

            numer = th.sum(act * g, 1)  + c + self.slack
            denomin = th.sum(g * g, 1) + 1e-8
            # # (B,1,A)x(B,A,1) -> (B,1,1) -> (B,)
            # numer = th.bmm(g_i.unsqueeze(1),
            #                   act.unsqueeze(2)).view(-1) + c_i + self.slack[i]
            # denomin = th.bmm(g_i.unsqueeze(1),
            #                     g_i.unsqueeze(2)).view(-1) + 1e-8
            # Equation (5) from Dalal 2018.
            mult = F.relu(numer / denomin)  # (B,)
            multipliers.append(mult)
        multipliers = th.stack(multipliers, -1)  # (B,C)
        # Check assumption on at most 1 active constraint
        # - as mentioned in the original paper, this simple, analytical solution of the safety layer only holds
        # with this assumption; otherwise resort to a differentiable layer for solving constrained optimization,
        # e.g. OptLayer or the differentiable MPC works; or alternatively combine multiple constraints to a single one.
        # - if the assumption is not satisfied, the layer will try to address the worst violation from the
        # the largest lagrange variable (with the use of `topk(..., 1)`)
        # - to check the assumption, check for each step in batch if |{i | \lambda_i > 0}| <= 1
        if float(th.gt(multipliers, 0).float().sum()) > multipliers.shape[0]:
            warnings.warn("""Assumption of at most 1 active constraint per step is violated in the current batch, 
                the filtered action will alleviate the worst violation but do not guarantee 
                satisfaction of all constraints, are you sure to proceed?""")
        # Calculate correction, equation (6) from Dalal 2018.
        max_mult, max_idx = th.topk(multipliers, 1, dim=-1)  # (B,1)
        max_idx = max_idx.view(-1).tolist()  # []_B
        # [(A,)]_B -> (B,A)
        max_g = th.stack([g[max_i][i] for i, max_i in enumerate(max_idx)])
        # (B,1) x (B,A) -> (B,A)
        correction = max_mult * max_g
        action_new = act - correction
        if len(action_new) == 1:
            action_new = action_new.view(-1)
        return action_new


class ConstraintBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros(alg_core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(alg_core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_next_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def __len__(self):
        return self.size

    def push_batch(self, obs, acs, rews, next_obs, dones):
        assert obs.shape[0] == acs.shape[0] == dones.shape[0]
        assert rews.shape[0] == next_obs.shape[0]
        batch_size = obs.shape[0]

        if (self.ptr + batch_size) <= self.max_size:
            _slice = slice(self.ptr, self.ptr + batch_size)
            self.obs_buf[_slice] = obs
            self.ptr = (self.ptr + batch_size) % self.max_size
            self.size = int(min(self.size + batch_size, self.max_size))
        else:
            diff = self.max_size - self.ptr
            slc1 = slice(0, diff)
            self.push_batch(obs[slc1], acs[slc1], rews[slc1], next_obs[slc1],
                            dones[slc1])

            slc2 = slice(diff, batch_size)
            self.push_batch(obs[slc2], acs[slc2], rews[slc2], next_obs[slc2],
                            dones[slc2])

    def store(self, obs, act, cost, next_cost):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.cost_buf[self.ptr] = cost
        self.cost_next_buf[self.ptr] = next_cost
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = int(min(self.size + 1, self.max_size))

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs],
                     c=self.cost_buf[idxs],
                     c_next=self.cost_next_buf[idxs])
        return {k: th.as_tensor(v, dtype=th.float32) #.to(device=self.device)
                for k, v in batch.items()}


class SafeExplorationPPO(ppo.ProximalPolicyOptimizationAlgorithm):
    def __init__(
            self,
            alg='safexppo',
            clip_ratio: float = 0.2,
            constraint_lr: float = 0.0001,  # fixme sven: set correct value
            constraint_slack: float = 0.1,  # fixme sven: set correct value
            constraint_hidden_dim: int = 10,  # fixme sven: set correct value
            init_exploration_steps: int = 2048,
            **kwargs
    ):
        super().__init__(
            alg=alg,
            clip_ratio=clip_ratio,
            **kwargs)

        self.constraint_lr = constraint_lr
        self.constraint_slack = constraint_slack
        self.constraint_hidden_dim = constraint_hidden_dim
        self.num_constraints = 1  #  fixme sven: get from env
        self.init_exploration_steps = init_exploration_steps

        self.safety_layer = SafetyLayer(
            self.env.observation_space,
            self.env.action_space,
            hidden_dim=self.constraint_hidden_dim,
            num_constraints=self.num_constraints,
            lr=self.constraint_lr,
            slack=self.constraint_slack
        )

        self.num_constraint_samples = 1000
        self.constraint_buffer = ConstraintBuffer(
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=int(self.num_constraint_samples),
        )

        self.global_step = 0

    def collect_data_and_update_constraint_model(self):
        """Uses random policy to collect data for pre-training constriant models.
        """
        step = 0
        env = copy.deepcopy(self.env)
        obs, _ = env.reset()
        c = 0.
        for step in range(self.num_constraint_samples):
            action = env.action_space.sample()
            obs_next, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            c_next = info.get('cost', 0.0)
            self.constraint_buffer.store(obs, action, c, c_next)
            if done:
                obs_next, _ = env.reset()
                c = 0.0
            obs = obs_next
            c = c_next
        
        for _ in range(16):
            batch = self.constraint_buffer.sample_batch(64)
            self.safety_layer.update(batch)

    def learn(self) -> tuple:
        # Main loop: collect experience in env and update/log each epoch
        for self.epoch in range(self.epochs):
            self.collect_data_and_update_constraint_model()
            self.learn_one_epoch()

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.ac, self.env

    def roll_out(self) -> None:
        """collect data and store to experience buffer."""
        o, _ = self.env.reset()
        ep_ret, ep_costs, ep_len, violations = 0., 0., 0, 0
        d = False
        c = 0.
        lstm_state = self.ac.get_initial_lstm_state()

        for t in range(self.local_steps_per_epoch):
            a, v, cv, logp, lstm_state = self.ac.step(
                th.as_tensor(o, dtype=th.float32),
                lstm_state,
                th.as_tensor(d, dtype=th.float32)
            )
            a = self.safety_layer.get_safe_action(o, a, c)
            next_o, r, terminated, truncated, info = self.env.step(a)
            c = info.get('cost', 0.)
            ep_ret += r
            ep_costs += c
            ep_len += 1
            self.global_step += mpi_tools.num_procs()
            if not self.env.observation_space.contains(next_o):
                violations += 1

            # save and log
            # Notes:
            #   - raw observations are stored to buffer (later transformed)
            #   - reward scaling is performed in buf
            self.buf.store(
                obs=o, act=a, rew=r, done=float(d), val=v, logp=logp,
                cost=c, cost_val=cv
            )
            if self.use_cost_value_function:
                self.logger.store(**{
                    'Values/V': v,
                    'Values/C': cv})
            else:
                self.logger.store(**{'Values/V': v})
            o = next_o

            done = terminated or truncated
            epoch_ended = t == self.local_steps_per_epoch - 1

            if done or epoch_ended:
                if truncated or epoch_ended:
                    _, v, cv, _, lstm_state = self.ac(
                        th.as_tensor(o, dtype=th.float32),
                        lstm_state,
                        th.as_tensor(d, dtype=th.float32)
                    )
                else:
                    v, cv = 0., 0.
                self.buf.finish_path(v, cv)
                if terminated:  # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len,
                                      EpCosts=ep_costs)
                o, _ = self.env.reset()
                ep_ret, ep_costs, ep_len = 0., 0., 0

        self.total_constraint_violations += mpi_tools.mpi_sum(violations)


def get_alg(env_id, **kwargs) -> SafeExplorationPPO:
    return SafeExplorationPPO(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='safexppo', env_id=env_id)
    defaults.update(**kwargs)
    alg = SafeExplorationPPO(
        env_id=env_id,
        **defaults
    )

    ac, env = alg.learn()

    return ac, env

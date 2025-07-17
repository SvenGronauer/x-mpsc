"""Safety Q-functions for reinforcement learning (SQRL)


See Paper:
Learning to be Safe: Deep RL with a Safety Critic

some parts are taken from:
https://sites.google.com/berkeley.edu/recovery-rl/
and
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
"""
import copy
import glob
import os
from typing import Optional

import ffmpeg
from copy import deepcopy
import itertools
import gymnasium as gym
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# local imports
import x_mpsc.common.mpi_tools as mpi
from x_mpsc.algs.mbpo.env_sampler import EnvSampler
from x_mpsc.algs.sqrl.qrisk import QRiskWrapper
from x_mpsc.common import loggers
from x_mpsc.algs import core
from x_mpsc.common import utils
from x_mpsc.algs.sqrl.buffer import ReplayBuffer
import x_mpsc.algs.utils as U
from x_mpsc.algs.sac.actor_critic import MLPActorCritic


class SafetyQFunctionRL(core.OffPolicyGradientAlgorithm):
    def __init__(
            self,
            env_id,
            logger_kwargs: dict,
            actor='mlp',  # meaningless at the moment
            ac_kwargs=dict(),
            alg='sqrl',
            alpha=0.2,
            automatic_entropy_tuning: bool = True,
            batch_size=512,
            buffer_size=int(1e6),
            check_freq: int = 25,
            epochs=100,
            eps_safe: float = 0.1,
            gamma=0.99,
            gamma_safe: float = 0.7,
            init_exploration_steps=8000,
            lr=1e-3,
            mini_batch_size=128,
            nu: float = 0.01,  # penalty term in Lagrangian objective
            polyak=0.995,
            save_freq=1,
            seed=0,
            update_every=64,
            verbose: bool = True,
            video_freq: int = -1,  # set to positive integer for video recording
            **kwargs  # use to log parameters from child classes
    ):
        assert 0 < polyak < 1
        self.params = locals()

        self.alg = alg
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.batch_size = batch_size
        self.check_freq = check_freq
        self.local_mini_batch_size = mini_batch_size // mpi.num_procs()
        self.local_batch_size = batch_size // mpi.num_procs()
        self.epoch = 0
        self.eps_safe = eps_safe
        self.epochs = epochs
        self.gamma = gamma
        self.init_exploration_steps = init_exploration_steps // mpi.num_procs()
        self.nu = nu
        self.polyak = polyak
        self.save_freq = save_freq
        self.update_every = update_every // mpi.num_procs()
        self.video_freq = video_freq

        # Note: NEW: call gym.make with **kwargs (to allow customization)
        if isinstance(env_id, str):
            self.env = gym.make(env_id, **kwargs)
        else:
            self.env = env_id
            self.params.pop('env_id')  # already instantiated envs cause errors

        self.act_limit = self.env.action_space.high[0]
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        self.device = U.get_device()
        self.torchify = lambda x: torch.as_tensor(x, dtype=torch.float32).to(
            self.device)

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]
        # ==== Call assertions....
        self._sanity_checks()

        # === Set up logger and save configuration to disk
        self.logger_kwargs = logger_kwargs
        self.logger = self._init_logger()
        self.logger.save_config(self.params)
        # save environment settings to disk
        self.logger.save_env_config(env=self.env)
        loggers.set_level(loggers.INFO)

        # === Seeding
        seed += 10000 * mpi.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed=seed)
        self.env_sampler = EnvSampler(self.env)

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            self.env.observation_space, self.env.action_space, ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        self.sc = QRiskWrapper(
            self.env.observation_space, self.env.action_space, ac_kwargs,
            gamma_safe=gamma_safe,
            lr=lr)

        self.target_entropy = -torch.prod(
            torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_nu = nn.Parameter(torch.log(self.torchify(nu)), requires_grad=True) #, device=self.device)
        self.nu_optim = Adam([self.log_nu], lr=0.1 * lr)

        self.log_alpha = nn.Parameter(self.torchify(0), requires_grad=True)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        self.freeze(self.ac_targ.parameters())

        # Initial ensemble model
        state_size = np.prod(self.env.observation_space.shape)
        action_size = np.prod(self.env.action_space.shape)

        nx = self.env.observation_space.shape[0]

        # === set up MPI specifics and sync parameters across all processes
        self._init_mpi()

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters()
        )
        # Experience buffer
        self.buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=buffer_size // mpi.num_procs()
        )

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in
                           [self.ac.pi, self.ac.q1, self.ac.q2])
        loggers.info('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up model saving
        self.logger.setup_actor_critic_saver(self.ac)
        # self.logger.setup_ensemble_model_saver(self.ensemble_dynamics_model.ensemble_model)
        self.logger.actor_critic_save()
        # self.logger.ensemble_model_save()

        # setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.total_constraint_violations = 0
        self.logger.info('Done with initialization.')

    def _init_logger(self):
        # pop to avoid self object errors
        self.params.pop('self')
        # move nested kwargs to highest dict level
        if 'kwargs' in self.params:
            self.params.update(**self.params.pop('kwargs'))
        logger = loggers.EpochLogger(**self.logger_kwargs)
        return logger

    def _init_mpi(self) -> None:
        """ Initialize MPI specifics

        Returns
        -------

        """
        if mpi.num_procs() > 1:
            loggers.info(f'Started MPI with {mpi.num_procs()} processes.')
            # Avoid slowdowns from PyTorch + MPI combo.
            mpi.setup_torch_for_mpi()
            dt = time.time()
            loggers.info('Sync actor critic parameters...')
            # Sync params across cores: only once necessary, grads are averaged!
            mpi.sync_params(self.sc)
            mpi.sync_params(self.ac)
            self.ac_targ = deepcopy(self.ac)
            loggers.info(f'Done! (took {time.time() - dt:0.3f} sec.)')
            self.check_distributed_parameters()

    def _sanity_checks(self):
        assert self.local_mini_batch_size > 0, f"Please increase batch size"
        assert isinstance(self.env, gym.Env), 'Env is not the expected type.'

    def algorithm_specific_logs(self):
        """ Use this method to collect log information. """
        pass

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if mpi.num_procs() > 1:
            loggers.info('Check if distributed parameters are synchronous')
            modules = {'Policy': self.ac.pi, 'Q1': self.ac.q1, 'Q2': self.ac.q2,
                       'SQ1': self.sc.q1, 'SQ2': self.sc.q2}
            for key, module in modules.items():
                flat_params = U.get_flat_params_from(module).numpy()
                global_min = mpi.mpi_min(np.sum(flat_params))
                global_max = mpi.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

            single_params = {'nu': self.log_nu, 'alpha': self.log_alpha}
            for key, param in single_params.items():
                global_min = mpi.mpi_min(np.sum(param.detach().numpy()))
                global_max = mpi.mpi_max(np.sum(param.detach().numpy()))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], \
                         data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (
                        q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = {'Values/Q1': q1.detach().numpy(),
                  'Values/Q2': q2.detach().numpy()}
        return loss_q, q_info

    def compute_nu_loss(self, data):
        with torch.no_grad():
            o = data['obs']
            pi, logp_pi = self.ac.pi(o)
            sqf1_pi = self.sc.q1(o, pi)
            sqf2_pi = self.sc.q2(o, pi)
            max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)
        nu_loss = (self.log_nu * (self.eps_safe - max_sqf_pi).detach()).mean()
        return nu_loss

    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        with torch.no_grad():
            sqf1_pi = self.sc.q1(o, pi)
            sqf2_pi = self.sc.q2(o, pi)
            max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)
        safe_loss = self.nu * (max_sqf_pi - self.eps_safe)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi + safe_loss).mean()

        # Useful info for logging
        pi_info = {'Values/LogPi': logp_pi.detach().numpy()}
        return loss_pi, pi_info

    @staticmethod
    def freeze(params):
        for p in params:
            p.requires_grad = False

    @property
    def in_warm_up(self) -> bool:
        return len(self.buffer) < self.init_exploration_steps

    def learn(self) -> tuple:
        r"""Main loop: collect experience in env and update/log each epoch."""
        for self.epoch in range(self.epochs):
            self.epoch_time = time.time()
            is_last_epoch = self.epoch == self.epochs - 1
            # Note: update function is called during rollouts!
            self.roll_out(batch_size=self.local_batch_size)
            # Save (or print) information about epoch
            self.log(self.epoch)

            # Check if all models own the same parameter values
            if self.epoch % self.check_freq == 0:
                self.check_distributed_parameters()

            # Save model to disk
            if is_last_epoch or self.epoch % self.save_freq == 0:
                self.logger.save_state(state_dict={}, itr=None)

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.ac, self.env

    def log(self, epoch):
        # Log info about epoch
        N = mpi.num_procs()
        total_env_steps = (epoch + 1) * self.batch_size
        fps = N * self.batch_size / (time.time() - self.epoch_time)

        self.logger.log_tabular('Epoch', epoch + 1)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=True)
        # self.logger.log_tabular('Values/Q1', min_and_max=True)
        # self.logger.log_tabular('Values/Q2', min_and_max=True)
        # self.logger.log_tabular('Values/LogPi', std=False)
        # self.logger.log_tabular('Loss/Pi', std=False)
        # self.logger.log_tabular('Loss/Q', std=False)

        self.logger.log_tabular('Misc/TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('Safety/Violations',
                                self.total_constraint_violations)
        self.logger.log_tabular('Misc/InWarmUp', float(self.in_warm_up))
        self.logger.log_tabular('Misc/Time', int(time.time() - self.start_time))
        self.logger.log_tabular('Misc/FPS', int(fps))
        self.logger.dump_tabular()

    def get_action(
            self,
            o: np.ndarray,
            deterministic: bool
    ) -> np.ndarray:
        if self.in_warm_up:
            a = self.env.action_space.sample()
        else:
            safe_samples = 100
            obs_batch = self.torchify(o).repeat(safe_samples, 1)
            with torch.no_grad():
                acs_batch, log_pi = self.ac.pi(obs_batch)
                max_qf_constraints = self.sc.get_value(obs_batch, acs_batch)

            thresh_idxs = (max_qf_constraints <= self.eps_safe).nonzero()[:, 0]

            logits = log_pi[thresh_idxs].flatten()

            if list(logits.size())[0] == 0:
                min_q_value_idx = torch.argmin(max_qf_constraints)
                a = acs_batch[min_q_value_idx, :].unsqueeze(0)
            else:
                prob_dist = torch.distributions.Categorical(logits=logits)
                sampled_idx = prob_dist.sample()
                a = acs_batch[sampled_idx].squeeze().cpu().numpy()

        return a

    def roll_out(self, batch_size: int):
        r"""Rollout and store to buffer."""
        violations = 0
        for t in range(batch_size):

            o = self.env_sampler.obs.copy()
            a = self.get_action(o, deterministic=False)
            next_o, r, done, info = self.env_sampler.step(a)
            con = info.get('cost', 0)
            self.buffer.store(o, a, r, next_o, done, con)

            if not self.env.observation_space.contains(next_o) and not self.in_warm_up:
                violations += 1

            # Update handling
            if t % self.update_every == 0:
                desc = f"INFO: \tTrain actor-critic"
                msg = f'\nSQRL: \tepoch: {self.epoch + 1}/{self.epochs} ' \
                      f'| Local freq: {self.update_every} batch step: {t}/{self.local_batch_size}'
                if mpi.is_root_process() and loggers.MAX_LEVEL >= loggers.INFO:
                    print(loggers.colorize(msg, color='cyan'))
                d = not mpi.is_root_process() or loggers.MAX_LEVEL >= loggers.INFO
                num_updates = self.update_every * mpi.num_procs()
                with tqdm(total=num_updates, desc=desc, ncols=80, disable=d) as pbar:
                    for _ in range(num_updates):
                        batchsize = int(min(4, self.local_mini_batch_size))
                        batch = self.buffer.sample_batch(batchsize)
                        self.update(data=batch)
                        pbar.update()

        self.logger.store(EpRet=self.env_sampler.average_trajectory_returns,
                          EpLen=self.env_sampler.average_trajectory_lengths)
        num_violations = mpi.mpi_sum(violations)
        self.total_constraint_violations += num_violations
        # if len(self.buffer) <= self.init_exploration_steps:
        #     # add zero values to prevent logging errors during warm-up
        #     self.logger.store(**{
        #         'Values/Q1': 0, 'Values/Q2': 0, 'Values/LogPi': 0,
        #         'Loss/Q': 0, 'Loss/Pi': 0,
        #     })

    @staticmethod
    def unfreeze(params):
        for p in params:
            p.requires_grad = True

    def update(self, data: dict):
        r"""Update Q functions and policy."""
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        mpi.mpi_avg_grads(self.ac.q1)  # average grads across MPI processes
        mpi.mpi_avg_grads(self.ac.q2)  # average grads across MPI processes
        self.q_optimizer.step()

        # Record things
        q_info['Loss/Q'] = loss_q.item()
        # self.logger.store(**q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.freeze(self.q_params)

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        mpi.mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        if self.in_warm_up:
            self.sc.update_parameters(data, self.ac.pi)
        else:
            nu_loss = self.compute_nu_loss(data)
            self.nu_optim.zero_grad()
            nu_loss.backward()
            self.log_nu.grad.copy_(mpi.mpi_avg(self.log_nu.grad))
            self.nu_optim.step()
            self.nu = self.log_nu.exp()

            if self.automatic_entropy_tuning:
                pi, logp_pi = self.ac.pi(data['obs'])
                alpha_loss = -(self.log_alpha *
                               (logp_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.log_alpha.grad.copy_(mpi.mpi_avg(self.log_alpha.grad))
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.unfreeze(self.q_params)

        # Record things
        pi_info['Loss/Pi'] = loss_pi.item()
        # self.logger.store(**pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


def get_alg(env_id, **kwargs) -> SafetyQFunctionRL:
    return SafetyQFunctionRL(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='sqrl', env_id=env_id)
    defaults.update(**kwargs)
    alg = SafetyQFunctionRL(
        env_id=env_id,
        **defaults
    )

    ac, env = alg.learn()

    return ac, env

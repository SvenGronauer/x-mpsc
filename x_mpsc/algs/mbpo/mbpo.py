""" PyTorch implementation of MBPO Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    06.12.2022

based on the Spinning Up implementation:
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac
"""
from __future__ import annotations

from copy import deepcopy
import itertools
from typing import Dict, Optional, Tuple

import gymnasium as gym
import time
import glob
import os
import ffmpeg
from tqdm import tqdm
import numpy as np
import torch as th
from torch.optim import Adam

# local imports
import x_mpsc.common.mpi_tools as mpi
from x_mpsc.algs.mbpo.env_sampler import EnvSampler
from x_mpsc.common import loggers
from x_mpsc.algs import core
from x_mpsc.mpsc import EnsembleMPSC
from x_mpsc.common import utils
import x_mpsc.algs.utils as U
from x_mpsc.algs.mbpo.buffer import ReplayBuffer
from x_mpsc.algs.mbpo.env_wrapper import PredictEnv
from x_mpsc.models import DynamicsModel
from x_mpsc.algs.utils import get_device
from x_mpsc.algs.sac.actor_critic import MLPActorCritic
from x_mpsc.envs.simple_pendulum.pendulum import SimplePendulumEnv
from x_mpsc.envs.twolinkarm.twolinkarm import TwoLinkArmEnv
from x_mpsc.envs.cartpole.cartpole import CartPoleEnv
from x_mpsc.mpsc.wrappers import EnsembleModelCasadiWrapper
from x_mpsc.algs.terminal_set import TerminalSet


def to_tensor(x: np.ndarray):
    return th.as_tensor(x, dtype=th.float32)


class ModelBasedPolicyOptmizationAlg(core.OffPolicyGradientAlgorithm):
    def __init__(
            self,
            env_id,
            logger_kwargs: dict,
            actor='mlp',  # meaningless at the moment
            ac_kwargs=dict(),
            alg='mbpo',
            alpha=0.2,
            batch_size=1024,
            buffer_size=1024000,
            check_freq: int = 2,
            ensemble_hiddens: tuple = (20, 20),
            epochs=250,
            gamma=0.99,
            grow_factor: float = 0.2,  # growing speed of terminal set
            delay_factor: int = 3,  # growing speed of terminal set
            lr=3e-4,
            mini_batch_size=128,

            model_retain_epochs: int = 20,
            model_train_freq: int = 256,
            mpsc_feedback_factor: float = 0.5,
            mpsc_horizon: int = 15,
            ensemble_size: int = 5,
            num_model_rollouts: int = 200,  # M model rollouts in each timestep

            real_ratio: float = 0.1,
            rollout_max_length: int = 10,
            rollout_min_length: int = 1,
            rollout_max_epoch: int = 100,
            rollout_min_epoch: int = 0,

            policy_train_iterations: int = 20,  # G policy updates per step
            pretrain_policy: bool = True,
            polyak=0.995,
            save_freq=1,
            seed=0,
            init_exploration_steps=5000,
            use_mpsc: bool = False,
            use_prior_model: bool = False,
            update_every=64,
            video_freq: int = -1,  # set to positive for video recording
            warm_up_ensemble_train_epochs: int = 100,
            **kwargs  # use to log parameters from child classes
    ):
        assert 0 < polyak < 1
        self.params = locals()

        self.alg = alg
        self.alpha = alpha
        self.batch_size = batch_size
        self.check_freq = check_freq
        self.delay_factor = delay_factor
        self.device = get_device()
        self.local_mini_batch_size = mini_batch_size // mpi.num_procs()
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.gamma = gamma
        self.local_batch_size = batch_size // mpi.num_procs()
        self.model_horizon = rollout_min_length
        self.model_train_freq = model_train_freq // mpi.num_procs()
        self.model_retain_epochs = model_retain_epochs
        self.mpsc_feedback_factor = mpsc_feedback_factor
        self.mpsc_horizon = mpsc_horizon
        self.mini_batch_size = mini_batch_size
        self.num_model_rollouts = num_model_rollouts
        self.policy_train_iterations = policy_train_iterations
        self.polyak = polyak
        self.pretrain_policy = pretrain_policy
        self.real_ratio = real_ratio

        self.rollout_max_length = rollout_max_length
        self.rollout_min_length = rollout_min_length
        self.rollout_max_epoch = rollout_max_epoch
        self.rollout_min_epoch = rollout_min_epoch

        self.save_freq = save_freq
        self.init_exploration_steps = init_exploration_steps
        self.update_every = update_every // mpi.num_procs()
        self.use_mpsc = use_mpsc
        self.use_prior_model = use_prior_model
        self.in_warm_up = True
        self.video_freq = video_freq
        self.warm_up_ensemble_train_epochs = warm_up_ensemble_train_epochs

        # Note: NEW: call gym.make with **kwargs (to allow customization)
        if isinstance(env_id, str):
            self.env = gym.make(env_id, **kwargs)
        else:
            self.env = env_id
            self.params.pop('env_id')  # already instantiated envs cause errors
        self.eval_env = gym.make(env_id, **kwargs)
        self.act_limit = self.env.action_space.high[0]
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

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

        # === Seeding
        self.seed = seed
        seed += 10000 * mpi.proc_id()
        th.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed=seed)

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            self.env.observation_space, self.env.action_space, ac_kwargs)
        self.ac_targ = deepcopy(self.ac)
        self.ac.to(device=self.device)
        self.ac_targ.to(device=self.device)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        self.freeze(self.ac_targ.parameters())

        # Initial ensemble model
        self.dynamics_model = DynamicsModel(
            self.env,
            ensemble_size,
            elite_size=ensemble_size,
            hidden_sizes=ensemble_hiddens,
            use_prior_model=use_prior_model,
            use_decay=True
        )

        # === set up MPI specifics and sync parameters across all processes
        self._init_mpi()

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.ac.q1.parameters(), self.ac.q2.parameters()
        )
        # Experience buffer
        self.real_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=buffer_size//mpi.num_procs()
        )
        self.virtual_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=self._new_virtual_buf_size
        )

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        loggers.info( '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        self.terminal_set = TerminalSet(delay_factor=delay_factor)

        # Set up model saving
        self.logger.setup_actor_critic_saver(self.ac)
        self.logger.setup_ensemble_model_saver(self.dynamics_model.ensemble_model)
        self.logger.actor_critic_save()
        self.logger.ensemble_model_save()
        self.logger.save_state(state_dict=self.state_dict, itr=None)
        
        self.mpsc = self._setup_mpsc()

        self.virtual_env = PredictEnv(self.dynamics_model, env_id)
        self.env_sampler = EnvSampler(self.env)

        # setup statistics
        self.epoch = 0
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.global_time_step = 0
        self.local_time_step = 0
        self.total_constraint_violations = 0
        self.total_certification_failures = 0
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
            mpi.sync_params(self.ac)
            self.ac_targ = deepcopy(self.ac)
            mpi.sync_params(self.dynamics_model.ensemble_model)
            loggers.info(f'Done! (took {time.time()-dt:0.3f} sec.)')
            self.check_distributed_parameters()

    @property
    def _new_virtual_buf_size(self):
        multiplier = self.local_batch_size // self.model_train_freq
        new_size = multiplier * self.model_retain_epochs * self.rollout_batch_size
        assert new_size > 0
        return new_size

    def _resize_virtual_buffer(self) -> ReplayBuffer:
        new_size = self._new_virtual_buf_size
        assert new_size >= len(self.virtual_buffer), "Buffer can only grow!"
        batch = self.virtual_buffer.return_all()
        new_buffer = ReplayBuffer(
            self.virtual_buffer.obs_dim, self.virtual_buffer.act_dim, new_size)
        new_buffer.push_batch(*batch)
        msg = f"New virtual buffer with size: {new_size}"
        loggers.info(loggers.colorize(msg, color='yellow'))
        return new_buffer
    
    def _setup_mpsc(self) -> Optional[EnsembleMPSC]:
        if self.use_mpsc:
            return EnsembleMPSC(
                env=self.env,
                dynamics_model=self.dynamics_model,
                horizon=self.mpsc_horizon,
                terminal_set=self.terminal_set,
                feedback_factor=self.mpsc_feedback_factor,
            )
        return None

    def estimate_terminal_set(self):

        if not self.use_mpsc:
            return
        loggers.info("Estimate terminal set...")
        obs = self.real_buffer.return_all_obs_where_mpsc_was_feasible()
        self.terminal_set.solve(
            data=obs,
            space=self.env.observation_space
        )
        # synchronize terminal sets
        # mpi.broadcast(self.terminal_set.ellipse.Q)
        # mpi.broadcast(self.terminal_set.ellipse.c)

    def fit_input_scaler(self):
        if len(self.real_buffer) == 0:
            return
        loggers.debug(f"Fit input scaling factors...")
        obs, acs, *_ = self.real_buffer.return_all()
        states_and_actions = np.concatenate((obs, acs), axis=-1)
        self.dynamics_model.fit_scaler(states_and_actions)

    def set_model_horizon(
            self,
            epoch
    ) -> None:
        loggers.info(f"current model horizon: {self.model_horizon}")
        new_model_horizon = int(
            min(max(self.rollout_min_length + (epoch - self.rollout_min_epoch)
                    / (self.rollout_max_epoch - self.rollout_min_epoch) * (
                                self.rollout_max_length - self.rollout_min_length),
                    self.rollout_min_length), self.rollout_max_length))

        if self.model_horizon != new_model_horizon:
            self.virtual_buffer = self._resize_virtual_buffer()
            msg = f"New model rollout length: {new_model_horizon}"
            loggers.info(loggers.colorize(msg, color='yellow'))
            self.model_horizon = new_model_horizon

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
                       'ensemble': self.dynamics_model.ensemble_model}
            for key, module in modules.items():
                flat_params = U.get_flat_params_from(module).numpy()
                global_min = mpi.mpi_min(np.sum(flat_params))
                global_max = mpi.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], \
                         data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with th.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = th.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = {
            'Values/Q1': q1.detach().cpu().numpy(),
            'Values/Q2': q2.detach().cpu().numpy(),
            'Values/Q_logp_a2': logp_a2.detach().cpu().numpy(),
            'Values/Q_target': q_pi_targ.detach().cpu().numpy(),
            'Values/Q_backup': backup.detach().cpu().numpy(),
            'Loss/Q': loss_q.item(),
        }
        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = th.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = {
            'Values/LogPi': logp_pi.detach().cpu().numpy(),
            'Loss/Pi': loss_pi.item(),
        }

        return loss_pi, pi_info

    @staticmethod
    def freeze(params):
        for p in params:
            p.requires_grad = False

    @staticmethod
    def unfreeze(params):
        for p in params:
            p.requires_grad = True

    @th.no_grad()
    def evaluate(self) -> None:
        r"""
        avg_reward_len = min(len(env_sampler.trajectory_returns), 5)
        avg_reward = sum(env_sampler.trajectory_returns[-avg_reward_len:]) / avg_reward_len
        loggers.info("Step Reward: " + str(self.global_time_step) + " " + str(env_sampler.trajectory_returns[-1]) + " " + str(avg_reward))
        print(self.global_time_step, env_sampler.trajectory_returns[-1], avg_reward)
        """
        loggers.info("Evaluate...")
        for _ in range(5):
            x, _ = self.eval_env.reset()
            eval_reward = 0
            done = False
            test_step = 0
            while (not done):
                a, _ = self.get_action(x, deterministic=True, certify=False)
                x, reward, terminated, truncated, info = self.eval_env.step(a)
                done = terminated or truncated
                eval_reward += reward
                test_step += 1
            self.logger.store(EvalRet=eval_reward)

    def get_action(
            self,
            o: np.ndarray,
            certify=True,
            deterministic=False
    ) -> Tuple[np.ndarray, bool]:
        feasible = True
        if self.in_warm_up:
            if hasattr(self.env, 'safe_controller'):
                a = self.env.safe_controller(o)
            else:
                a = self.env.action_space.sample()
            feasible = self.env_sampler.is_init_obs
        else:
            th_obs = th.as_tensor(o, dtype=th.float32).to(device=self.device)
            a = self.ac.act(th_obs, deterministic)
            if certify and self.use_mpsc:
                a = self.mpsc.solve(o, a)
                feasible = self.mpsc.feasible
        return a, feasible

    def learn(self) -> tuple:
        r"""Main loop: collect experience in env and update/log each epoch."""
        self.run_warmup()
        self.in_warm_up = False

        for self.epoch in range(self.epochs):
            self.estimate_terminal_set()
            self.record_video()
            self.epoch_time = time.time()
            self.set_model_horizon(epoch=self.epoch)
            self.fit_input_scaler()
            # Note: update function is called during rollouts!
            self.roll_out(batch_size=self.local_batch_size)
            self.evaluate()
            self.log(self.epoch)

            if self.epoch % self.check_freq == 0:
                self.check_distributed_parameters()

            is_last_epoch = self.epoch == self.epochs - 1
            if is_last_epoch or self.epoch % self.save_freq == 0:
                self.logger.save_state(state_dict=self.state_dict, itr=None)

        loggers.debug('Close Logger and associated files.')
        self.logger.close()
        return self.ac, self.env

    def log(self, epoch):
        # Log info about epoch
        fps = self.batch_size / (time.time()-self.epoch_time)

        self.logger.log_tabular('Epoch', epoch+1)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=False)
        self.logger.log_tabular("EvalRet", std=True)

        self.logger.log_tabular('Values/Q1', min_and_max=False)
        self.logger.log_tabular('Values/Q2', min_and_max=False)
        self.logger.log_tabular('Values/Q_backup', min_and_max=False)
        self.logger.log_tabular('Values/Q_target', min_and_max=False)
        self.logger.log_tabular('Values/Q_logp_a2', min_and_max=False)
        self.logger.log_tabular('Values/LogPi', std=False)

        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Q', std=False)

        self.logger.log_tabular('MBPO/horizon', self.model_horizon)

        if self.real_ratio < 1.0:
            self.logger.log_tabular("Eval/prediction_mse")
            self.logger.log_tabular("Train/mse_delta")
            self.logger.log_tabular("Train/epochs")

        self.logger.log_tabular('Safety/UseMPSC', float(self.use_mpsc))
        self.logger.log_tabular('Safety/Violations',
                                self.total_constraint_violations)
        self.logger.log_tabular('Safety/CertificationFailures',
                                self.total_certification_failures)

        self.logger.log_tabular('Misc/InWarmUp', float(self.in_warm_up))
        self.logger.log_tabular('Misc/TotalEnvSteps', self.global_time_step)
        self.logger.log_tabular('Misc/Seed', self.seed)
        self.logger.log_tabular('Misc/Time', int(time.time() - self.start_time))
        self.logger.log_tabular('Misc/FPS', int(fps))
        self.logger.dump_tabular()

    @property
    def is_time_to_update_policy(self) -> bool:
        return self.local_time_step % self.update_every == 0 \
               and not self.in_warm_up

    @property
    def is_time_to_update_model_ensemble(self) -> bool:
        return self.local_time_step % self.model_train_freq == 0 \
               and not self.in_warm_up \
               and len(self.real_buffer) > self.local_mini_batch_size

    def do_policy_pretraining(self):
        loggers.info(f"Do policy pre-training...")
        ts = time.time()
        criterion = th.nn.MSELoss()
        optimizer = Adam(self.ac.pi.parameters())
        for i in range(500):
            optimizer.zero_grad()
            data = self.real_buffer.sample_batch(batch_size=256)
            action_pred, _ = self.ac.pi(to_tensor(data['obs']), deterministic=True)
            loss = criterion(input=action_pred, target=data['act'])
            loss.backward()
            mpi.mpi_avg_grads(self.ac.pi.net)
            mpi.mpi_avg_grads(self.ac.pi.mu_layer)
            optimizer.step()

        self.ac_targ = deepcopy(self.ac)
        self.ac_targ.to(device=self.device)
        self.freeze(self.ac_targ.parameters())
        loggers.info(f'Done. (took: {time.time()-ts:0.3f}s)')

    def roll_out(self, batch_size: int):
        r"""Rollout >>one<< episode and store to buffer."""
        loggers.debug(f"Do Rollouts...")
        failures, violations = 0, 0
        if self.mpsc is not None and not self.in_warm_up:
            self.mpsc.setup_optimizer()

        for t in range(batch_size):
            o = self.env_sampler.obs.copy()
            a, feasible = self.get_action(o, deterministic=False)
            next_o, r, done, info = self.env_sampler.step(a)
            self.real_buffer.store(o, a, r, next_o, done, feasible)

            if info.get('cost', 0.0) > 0.0:  # Note: use costs insted of not env.observation_space.contains(x) since env_sampler return x=env.reset() and not the terminal state!
                violations += 1
            if self.mpsc is not None and self.mpsc.is_failure:
                failures += 1

            if self.is_time_to_update_policy:
                self.update()
            if self.is_time_to_update_model_ensemble:
                N = mpi.num_procs()
                msg = f'\nMBPO: \tepoch: {self.epoch + 1}/{self.epochs} ' \
                      f'| GLOBAL freq: {self.update_every*N} step: {self.global_time_step}) ' \
                      f'| Local freq: {self.update_every} batch step: {t}/{self.local_batch_size}'
                if not self.in_warm_up and mpi.is_root_process() :
                    print(loggers.colorize(msg, color='cyan'))

                self.train_model_ensemble()
                self.roll_out_model()
                if self.mpsc is not None:
                    self.mpsc.setup_optimizer()

            if self.in_warm_up:
                self.global_time_step += 1  # due to broadcasting of init data
            else:
                self.global_time_step += mpi.num_procs()
            self.local_time_step += 1

        if not self.in_warm_up:
            self.logger.store(EpRet=self.env_sampler.average_trajectory_returns,
                              EpLen=self.env_sampler.average_trajectory_lengths)
            self.total_constraint_violations += mpi.mpi_sum(violations)
            self.total_certification_failures += mpi.mpi_sum(failures)

    def record_video(self):
        if not (self.epoch % self.video_freq == 0 and self.video_freq > 0):
            loggers.debug(f"Skip video recording...")
            return
        # if self.mpsc is None:
        #     return
        loggers.info(f"Start recording video...")
        record_env = deepcopy(self.env)

        if isinstance(record_env.unwrapped, SimplePendulumEnv):
            from x_mpsc.envs.simple_pendulum.wrapper import SimplePendulumPlotWrapper
            record_env = SimplePendulumPlotWrapper(record_env)

        elif isinstance(record_env.unwrapped, TwoLinkArmEnv):
            from x_mpsc.envs.twolinkarm.wrapper import TwoLinkArmEnvPlotWrapper
            record_env = TwoLinkArmEnvPlotWrapper(record_env)

        elif isinstance(record_env.unwrapped, CartPoleEnv):
            from x_mpsc.envs.cartpole.wrapper import CartPolePlotWrapper
            record_env = CartPolePlotWrapper(record_env)
        else:
            return None
            raise ValueError(f"Unknown env: {record_env}")

        print(f"recording with env: {record_env}")

        x = record_env.reset()
        if self.use_mpsc:
            self.mpsc.terminal_set = self.terminal_set
            self.mpsc.setup_optimizer()
        pbar = tqdm(range(self.local_mini_batch_size), disable=not mpi.is_root_process())
        nx = self.env.observation_space.shape[0]
        nu = self.env.action_space.shape[0]
        wrapped_models = [
            EnsembleModelCasadiWrapper(
                self.dynamics_model, model_idx=m) for m in range(self.ensemble_size)
        ]

        for stage in pbar:
            mpi.broadcast(x)
            pbar.set_description(f"Recording video")
            a, feasible = self.get_action(x, deterministic=True, certify=False)
            u_learn = np.clip(a, self.env.action_space.low, self.env.action_space.high)
            if mpi.is_root_process():
                nu = record_env.action_space.shape[0]
                if self.use_mpsc:
                    u_safe = self.mpsc.solve(x, a)
                    feasible = self.mpsc.feasible
                    Us = self.mpsc.last_Us if self.mpsc.last_Us is not None else np.zeros(
                        (nu, self.mpsc.horizon - 1))
                else:
                    u_safe = a
                    Us = np.zeros((nu, self.mpsc_horizon - 1))
                record_env.plot_current_nominal_trajectory(
                    wrapped_models,
                    self.mpsc,
                    u_learn,
                    Us,
                    log_dir=self.logger.log_dir,
                    epoch=self.epoch,
                    iteration=stage,
                    terminal_set=self.terminal_set
                )
            else:
                u_safe = u_learn
            x, r, done, info = record_env.step(u_safe)
        pbar.close()

        if mpi.is_root_process():
            video_path = os.path.join(self.logger.log_dir, "videos")
            jpg_path = os.path.join(video_path, "*.jpg")

            video_file_path = os.path.join(video_path, f'ep-{self.epoch}-movie.mp4')
            try:
                ffmpeg.input(jpg_path, pattern_type='glob', framerate=10).output(video_file_path).run(quiet=True)
            except:
                loggers.error("Could not create video file.")

            for file in glob.glob(jpg_path):
                loggers.trace(f"delete: {file}")
                os.remove(file)
        record_env.close()

    @property
    def rollout_batch_size(self):
        return self.local_batch_size * self.num_model_rollouts

    @th.no_grad()
    def roll_out_model(self):
        if self.real_ratio >= 1.0:
            loggers.info(f"Skip model rollouts.")
            return
        total_rollouts = self.rollout_batch_size * self.model_horizon
        desc = f"INFO: \tRollout models (k={self.model_horizon}) (buf_size={len(self.virtual_buffer)})"

        with tqdm(total=total_rollouts, desc=desc, ncols=80, unit='pass', disable=not mpi.is_root_process()) as pbar:
            data = self.real_buffer.sample_batch(self.rollout_batch_size)
            states = data['obs'].cpu().numpy()
            for _ in range(self.model_horizon):
                th_obs = th.as_tensor(states, dtype=th.float32).to(device=self.device)
                actions = self.ac.act(th_obs, deterministic=False)
                next_states, rewards, dones, info = self.virtual_env.step(
                    states, actions)
                self.virtual_buffer.push_batch(
                    states, actions, rewards, next_states, dones)
                nonterm_mask = ~dones
                if nonterm_mask.sum() == 0:
                    break
                states = next_states[nonterm_mask]
                pbar.update(self.rollout_batch_size)

    def run_warmup(self):
        loggers.info(f"Run warmup")
        self.roll_out(
            batch_size=self.init_exploration_steps
        )
        if self.pretrain_policy and self.use_mpsc:
            self.do_policy_pretraining()
        self.real_buffer.synchronize()
        self.fit_input_scaler()
        self.train_model_ensemble()
        self.estimate_terminal_set()
        self.logger.save_state(state_dict=self.state_dict, itr=None)
        loggers.info(f"Done with warmup!")

    @property
    def state_dict(self) -> Dict:
        return {'terminal_set': self.terminal_set}

    def train_model_ensemble(self) -> None:
        loggers.debug(f"Train model ensemble...)")
        batch_size = int(min(32, len(self.real_buffer)))
        state, action, reward, next_state, done = self.real_buffer.return_all()
        if self.use_prior_model:
            f = self.dynamics_model.prior_dynamics_model.predict_next_state

            if state.ndim == 1:
                next_model_obs = f(state, action)
            else:
                next_model_obs = np.array(list(map(f, state, action)))
            # print(next_obs)
            labels = next_state - next_model_obs.reshape(state.shape)
        else:
            labels = next_state
        inputs = np.concatenate((state, action), axis=-1)

        max_epochs = self.warm_up_ensemble_train_epochs if self.in_warm_up else 1
        holdout_ratio = 0.0 if self.in_warm_up else 0.2
        diags = self.dynamics_model.train(
            inputs, labels, batch_size=batch_size, holdout_ratio=holdout_ratio,
            terminate_early=not self.in_warm_up,
            max_epochs=max_epochs
        )
        self.logger.store(**diags)

    def update(self):
        num_rollouts = self.update_every * self.policy_train_iterations * mpi.num_procs()
        desc = f"INFO: \tTrain actor-critic"
        with tqdm(total=num_rollouts, desc=desc, ncols=80, disable=not mpi.is_root_process()) as pbar:
            num_real_iters = int(self.real_ratio * self.policy_train_iterations)
            for _ in range(self.update_every * mpi.num_procs()):
                for i in range(self.policy_train_iterations):
                    low_virtual_data = len(self.virtual_buffer) < self.local_mini_batch_size
                    if (self.real_ratio > 0 and i < num_real_iters) or low_virtual_data:
                        data = self.real_buffer.sample_batch(self.local_mini_batch_size)
                    else:
                        data = self.virtual_buffer.sample_batch(self.local_mini_batch_size)
                    self.update_once(data)
                pbar.update(self.policy_train_iterations)

    def update_once(self, data: Dict):
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        mpi.mpi_avg_grads(self.ac.q1)  # average grads across MPI processes
        mpi.mpi_avg_grads(self.ac.q2)  # average grads across MPI processes
        self.q_optimizer.step()

        self.logger.store(**q_info)

        self.freeze(self.q_params)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        mpi.mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()
        self.unfreeze(self.q_params)

        self.logger.store(**pi_info)

        with th.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


def get_alg(env_id, **kwargs) -> core.Algorithm:
    return ModelBasedPolicyOptmizationAlg(
        env_id=env_id,
        **kwargs
    )


# compatible class to OpenAI Baselines learn functions
def learn(env_id, **kwargs) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='mbpo', env_id=env_id)
    defaults.update(**kwargs)
    alg = ModelBasedPolicyOptmizationAlg(
        env_id,
        **defaults
    )
    ac, env = alg.learn()
    return ac, env

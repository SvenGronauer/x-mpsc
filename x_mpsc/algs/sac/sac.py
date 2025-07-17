""" PyTorch implementation of Soft Actor Critic (SAC) Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    07.07.2022

based on the Spinning Up implementation:
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
from torch.optim import Adam

# local imports
import x_mpsc.common.mpi_tools as mpi
from x_mpsc.common import loggers
from x_mpsc.algs import core
from x_mpsc.common import utils
from x_mpsc.algs.sac.buffer import ReplayBuffer
from x_mpsc.algs.terminal_set import TerminalSet
import x_mpsc.algs.utils as U
from x_mpsc.models.ensemble import DynamicsModel
from x_mpsc.algs.sac.actor_critic import MLPActorCritic
from x_mpsc.mpsc import EnsembleMPSC


class SoftActorCriticAlgorithm(core.OffPolicyGradientAlgorithm):
    def __init__(
            self,
            env_id,
            logger_kwargs: dict,
            actor='mlp',  # meaningless at the moment
            ac_kwargs=dict(),
            alg='sac',
            alpha=0.2,
            batch_size=1000,
            buffer_size=int(1e6),
            check_freq: int = 25,
            ensemble_hiddens: tuple = (20, 20),
            ensemble_size: int = 5,
            epochs=100,
            gamma=0.99,
            init_exploration_steps=2000,
            lr=1e-3,
            mini_batch_size=128,
            num_elites: int = 3,
            polyak=0.995,
            polyak_terminal_set=0.95,
            delay_factor=3,
            prediction_horizon: int = 10,
            save_freq=1,
            seed=0,
            update_every=50,
            use_mpsc: bool = False,
            verbose: bool = True,
            video_freq: int = 5,  # set to positive integer for video recording
            weight_decay: float = 1e-4,
            **kwargs  # use to log parameters from child classes
    ):
        assert 0 < polyak < 1
        self.params = locals()

        self.alg = alg
        self.alpha = alpha
        self.batch_size = batch_size
        self.check_freq = check_freq
        self.local_mini_batch_size = mini_batch_size // mpi.num_procs()
        self.epoch = 0
        self.epochs = epochs
        self.gamma = gamma
        self.polyak = polyak
        self.polyak_terminal_set = polyak_terminal_set
        self.save_freq = save_freq
        self.init_exploration_steps = init_exploration_steps
        self.update_every = update_every
        self.use_mpsc = use_mpsc
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

        # Create actor-critic module and target networks
        self.ac = MLPActorCritic(
            self.env.observation_space, self.env.action_space, ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers
        # (only update via polyak averaging)
        self.freeze(self.ac_targ.parameters())

        # Initial ensemble model
        state_size = np.prod(self.env.observation_space.shape)
        action_size = np.prod(self.env.action_space.shape)
        self.ensemble_dynamics_model = DynamicsModel(
            ensemble_size,
            num_elites,
            state_size,
            action_size,
            hidden_sizes=ensemble_hiddens,
            use_decay=True
        )

        nx = self.env.observation_space.shape[0]
        self.terminal_set = TerminalSet(delay_factor=delay_factor)

        self.mpsc = self._setup_mpsc()

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
            size=buffer_size//mpi.num_procs()
        )

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        loggers.info( '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up model saving
        self.logger.setup_actor_critic_saver(self.ac)
        self.logger.setup_ensemble_model_saver(self.ensemble_dynamics_model.ensemble_model)
        self.logger.actor_critic_save()
        self.logger.ensemble_model_save()

        # setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
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
            mpi.sync_params(self.ensemble_dynamics_model.ensemble_model)
            loggers.info(f'Done! (took {time.time()-dt:0.3f} sec.)')
            self.check_distributed_parameters()

    def _sanity_checks(self):
        assert self.local_mini_batch_size > 0, f"Please increase batch size"
        assert isinstance(self.env, gym.Env), 'Env is not the expected type.'

    def _setup_mpsc(self) -> Optional[EnsembleMPSC]:
        if self.use_mpsc:
            return EnsembleMPSC(
                env=self.env,
                dynamics_model=self.ensemble_dynamics_model,
                horizon=10
            )
        return None

    def algorithm_specific_logs(self):
        """ Use this method to collect log information. """
        pass
        # self.logger.log_tabular('Loss/ModelLoss')
        # self.logger.log_tabular('Loss/ModelDelta')

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if mpi.num_procs() > 1:
            loggers.info('Check if distributed parameters are synchronous')
            modules = {f"ensemble": self.ensemble_dynamics_model.ensemble_model}
            modules.update({'Policy': self.ac.pi, 'Q1': self.ac.q1, 'Q2': self.ac.q2})
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
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = {'Values/Q1': q1.detach().numpy(), 'Values/Q2': q2.detach().numpy()}

        return loss_q, q_info

    def compute_loss_pi(self, data):
        o = data['obs']
        is_act_unsafe = data['act_is_unsafe']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        safety_mse = is_act_unsafe * torch.sum((pi - data['act_learn'])**2, 1)
        loss_pi = (self.alpha * logp_pi - q_pi - safety_mse).mean()

        # Useful info for logging
        pi_info = {'Values/LogPi': logp_pi.detach().numpy(),
                   'Loss/ActionRegression': safety_mse.mean().item()}
        return loss_pi, pi_info

    def estimate_terminal_set(self):

        # FIXME Sven: terminal set differs in each process!

        if not self.use_mpsc:
            return
        loggers.info("estimate terminal set...")
        data = self.buffer.get_all_safe_samples()
        self.terminal_set.solve(data=data, space=self.env.observation_space)

    def fit_input_scaler(self):
        if len(self.buffer) == 0 or not self.use_mpsc:
            return
        data = self.buffer.return_all()
        states_and_actions = np.concatenate((data['obs'], data['act']), axis=-1)
        self.ensemble_dynamics_model.fit_scaler(states_and_actions)

    @staticmethod
    def freeze(params):
        for p in params:
            p.requires_grad = False

    def get_action(self, o: np.ndarray, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def learn(self) -> tuple:
        # Main loop: collect experience in env and update/log each epoch
        for self.epoch in range(self.epochs):
            self.epoch_time = time.time()
            is_last_epoch = self.epoch == self.epochs - 1
            self.fit_input_scaler()
            # Note: update function is called during rollouts!
            self.roll_out()
            # Save (or print) information about epoch
            self.log(self.epoch)
            self.estimate_terminal_set()
            self.record_video()

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
        total_env_steps = (epoch + 1) * N * self.batch_size
        fps = N * self.batch_size / (time.time() - self.epoch_time)

        self.logger.log_tabular('Epoch', epoch + 1)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=True)
        self.logger.log_tabular('Values/Q1', min_and_max=True)
        self.logger.log_tabular('Values/Q2', min_and_max=True)
        self.logger.log_tabular('Values/LogPi', std=False)
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Q', std=False)
        self.logger.log_tabular('Loss/ActionRegression', std=False)

        if self.use_mpsc:
            self.logger.log_tabular("Eval/prediction_mse")
            self.logger.log_tabular("Train/mse_delta")
            self.logger.log_tabular("Train/epochs")

        self.logger.log_tabular('Total/EnvSteps', total_env_steps)
        self.logger.log_tabular('Total/Violations', self.total_constraint_violations)
        self.logger.log_tabular('Total/CertificationFailures', self.total_certification_failures)
        self.logger.log_tabular('Misc/InWarmUp', float(self.in_warm_up))
        self.logger.log_tabular('Misc/Time', int(time.time() - self.start_time))
        self.logger.log_tabular('Misc/FPS', int(fps))
        self.logger.dump_tabular()

    @property
    def in_warm_up(self):
        return True if len(self.buffer) < self.init_exploration_steps else False

    def record_video(self):

        #todo sven: implement video recording
        return None

    def roll_out(self):
        r"""Rollout >>one<< episode and store to buffer."""

        o, ep_ret, ep_len, violations, failures = self.env.reset(), 0., 0, 0, 0
        # pbar = tqdm(range(self.batch_size), disable=not mpi.is_root_process())
        if not self.in_warm_up and self.mpsc is not None:
            self.mpsc.setup_optimizer()
        for t in range(self.batch_size):
            # pbar.set_description(f"Rollout")
            a_learn = self.env.action_space.sample() if self.in_warm_up \
                else self.get_action(o, deterministic=False)
            if self.use_mpsc and not self.in_warm_up:
                dt = time.time()
                a = self.mpsc.solve(o, a_learn)
                if mpi.is_root_process():
                    delta = time.time() - dt
                    print(f"X-MPSC took:\t{delta:0.3f}s LOL")
                if self.mpsc.is_failure:
                    failures += 1
            else:
                a = a_learn

            next_o, r, terminal, truncated, info = self.env.step(a)
            ep_ret += r
            ep_len += 1
            if not self.env.observation_space.contains(next_o):
                violations += 1

            # Store experience to replay buffer
            is_safe = True if np.linalg.norm(a - a_learn) < 1e-4 and self.env.observation_space.contains(next_o) else False
            self.buffer.store(o, a, r, next_o, terminal, a_learn, is_safe)

            # Update handling
            if t % self.update_every == 0:
                self.train_model_ensemble()
                desc = f"INFO: \tTrain actor-critic"
                if not self.in_warm_up:
                    with tqdm(total=self.update_every, desc=desc, ncols=80, disable=not mpi.is_root_process()) as pbar:
                        for _ in range(self.update_every):
                            batch = self.buffer.sample_batch(self.local_mini_batch_size)
                            # update all: Q funcs, policy and model ensemble
                            self.update(data=batch)
                            pbar.update()

            o = next_o
            if truncated or terminal:
                 # only save EpRet / EpLen if trajectory finished
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, _ = self.env.reset()
                ep_ret, ep_len = 0., 0
        # pbar.close()
        self.total_constraint_violations += mpi.mpi_sum(violations)
        self.total_certification_failures += mpi.mpi_sum(failures)
        if self.in_warm_up or len(self.buffer) == self.init_exploration_steps:
            # add zero values to prevent logging errors during warm-up
            self.logger.store(**{
                'Values/Q1': 0, 'Values/Q2': 0, 'Values/LogPi': 0,
                'Loss/Q': 0, 'Loss/Pi': 0, 'Loss/ActionRegression': 0
            })

    def train_model_ensemble(self) -> None:
        if not self.use_mpsc:
            return
        loggers.debug(f"Train model ensemble...")
        batch_size = int(min(256, len(self.buffer)))
        batch = self.buffer.return_all(as_numpy=True)
        delta_state = batch['obs2'] - batch['obs']
        inputs = np.concatenate((batch['obs'], batch['act'] ), axis=-1)
        labels = delta_state
        diags = self.ensemble_dynamics_model.train(
            inputs, labels, batch_size=batch_size, holdout_ratio=0.2)
        self.logger.store(**diags)

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
        self.logger.store(**q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.freeze(self.q_params)

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        mpi.mpi_avg_grads(self.ac.pi)  # average grads across MPI processes
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        self.unfreeze(self.q_params)

        # Record things
        pi_info['Loss/Pi'] = loss_pi.item()
        self.logger.store(**pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


def get_alg(env_id, **kwargs) -> core.Algorithm:
    return SoftActorCriticAlgorithm(
        env_id=env_id,
        **kwargs
    )


# compatible class to OpenAI Baselines learn functions
def learn(env_id, **kwargs) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='sac', env_id=env_id)
    defaults.update(**kwargs)
    alg = SoftActorCriticAlgorithm(
        env_id,
        **defaults
    )
    ac, env = alg.learn()
    return ac, env

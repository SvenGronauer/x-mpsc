r"""Pytorch implementation of Probabilistic Ensembles With trajectory sampling. (PETS).

Chua et al.
Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models
32nd Conference on Neural Information Processing Systems (NIPS 2018)

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    19.09.2022
Updated:    18.01.2023
"""
import copy
from copy import deepcopy
import itertools
from typing import Dict, Optional

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
from x_mpsc.models.ensemble import DynamicsModel
from x_mpsc.algs.utils import get_device
from x_mpsc.envs.simple_pendulum.pendulum import SimplePendulumEnv
from x_mpsc.envs.twolinkarm.twolinkarm import TwoLinkArmEnv
from x_mpsc.envs.cartpole import CartPoleEnv
from x_mpsc.mpsc.wrappers import EnsembleModelCasadiWrapper
from x_mpsc.algs.terminal_set import TerminalSet
import x_mpsc.algs.pets.controller as control


class ProbabilisticEnsemblesTrajectorySampling(core.Algorithm):
    def __init__(
            self,
            env_id,
            logger_kwargs: dict,
            actor='mlp',  # meaningless at the moment
            ac_kwargs=dict(),
            alg='pets',
            # alpha=0.2,
            batch_size=1024,
            buffer_size=1024000,

            cem_alpha: int = 0.1,
            cem_max_iters: int = 5,
            cem_num_elites: int = 32,
            cem_pop_size: int = 256,
            cem_trajectory_length: int = 25,

            check_freq: int = 2,
            ensemble_hiddens: tuple = (20, 20),
            epochs=250,
            gamma=0.99,
            grow_factor: float = 0.2,  # growing speed of terminal set
            lr=3e-4,
            mini_batch_size=128,

            #model_retain_epochs: int = 20,
            model_train_freq: int = 256,
            mpsc_horizon: int = 15,
            num_elites: int = 3,
            ensemble_size: int = 5,  # todo: set to 7
            #num_model_rollouts: int = 200,  # M model rollouts in each timestep

            #real_ratio: float = 0.1,
            #rollout_max_length: int = 10,
            #rollout_min_length: int = 1,
            #rollout_max_epoch: int = 100,
            #rollout_min_epoch: int = 0,

            #policy_train_iterations: int = 20,  # G policy updates per step
            #polyak=0.995,
            save_freq=1,
            seed=0,
            init_exploration_steps=5000,
            use_mpsc: bool = False,
            update_every=64,
            video_freq: int = -1,  # set to positive for video recording
            **kwargs  # use to log parameters from child classes
    ):
        self.params = locals()

        self.alg = alg
        self.batch_size = batch_size
        self.check_freq = check_freq
        self.device = get_device()
        self.local_mini_batch_size = mini_batch_size // mpi.num_procs()
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.gamma = gamma
        self.local_batch_size = batch_size // mpi.num_procs()
        self.model_train_freq = model_train_freq // mpi.num_procs()
        self.mpsc_horizon = mpsc_horizon
        self.num_elites = num_elites
        self.mini_batch_size = mini_batch_size

        self.save_freq = save_freq
        self.init_exploration_steps = init_exploration_steps
        self.update_every = update_every // mpi.num_procs()
        self.use_mpsc = use_mpsc
        self.in_warm_up = True
        self.video_freq = video_freq

        # Note: NEW: call gym.make with **kwargs (to allow customization)
        if isinstance(env_id, str):
            self.env = gym.make(env_id, **kwargs)
        else:
            self.env = env_id
            self.params.pop('env_id')  # already instantiated envs cause errors
        self.eval_env = copy.deepcopy(self.env)
        self.act_limit = self.env.action_space.high[0]
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

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

        self.controller = control.CrossEntropyMethodController(
            model=self.ensemble_dynamics_model,
            env=self.env,
            alpha=cem_alpha,
            trajectory_length=cem_trajectory_length,
            pop_size=cem_pop_size,
            num_elites=cem_num_elites,
            max_iters=cem_max_iters,
        )

        # === set up MPI specifics and sync parameters across all processes
        self._init_mpi()

        # Experience buffer
        self.real_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=buffer_size//mpi.num_procs()
        )

        self.virtual_env = PredictEnv(self.ensemble_dynamics_model, env_id)
        self.env_sampler = EnvSampler(self.env)

        self.terminal_set = TerminalSet(delay_factor=3)

        # Set up model saving
        self.logger.setup_ensemble_model_saver(self.ensemble_dynamics_model.ensemble_model)
        self.logger.ensemble_model_save()
        self.logger.save_state(state_dict=self.state_dict, itr=None)
        
        self.mpsc = self._setup_mpsc()

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
            mpi.sync_params(self.ensemble_dynamics_model.ensemble_model)
            loggers.info(f'Done! (took {time.time()-dt:0.3f} sec.)')
            self.check_distributed_parameters()
    
    def _setup_mpsc(self) -> Optional[EnsembleMPSC]:
        if self.use_mpsc:
            return EnsembleMPSC(
                env=self.env,
                dynamics_model=self.ensemble_dynamics_model,
                horizon=self.mpsc_horizon,
                terminal_set=self.terminal_set
            )
        return None

    def estimate_terminal_set(self, use_polyak):

        # FIXME Sven: terminal set differs in each process!

        if not self.use_mpsc:
            return
        loggers.info("Estimate terminal set...")
        # fixme sven: use only sampels where MPSC found a solution
        obs, *_ = self.real_buffer.return_all()
        self.terminal_set.solve(data=obs, space=self.env.observation_space)

    def fit_input_scaler(self):
        if len(self.real_buffer) == 0:
            return
        loggers.debug(f"Fit input scaling factors...")
        obs, acs, *_ = self.real_buffer.return_all()
        states_and_actions = np.concatenate((obs, acs), axis=-1)
        self.ensemble_dynamics_model.fit_scaler(states_and_actions)

    def _sanity_checks(self):
        assert self.local_mini_batch_size > 0, f"Please increase batch size"
        assert isinstance(self.env, gym.Env), 'Env is not the expected type.'
        assert hasattr(self.env, "calculate_reward"), \
            'Missing: calculate_reward() method in env'

    def algorithm_specific_logs(self):
        """ Use this method to collect log information. """
        pass

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if mpi.num_procs() > 1:
            loggers.info('Check if distributed parameters are synchronous')
            modules = {'Policy': self.ac.pi, 'Q1': self.ac.q1, 'Q2': self.ac.q2,
                       'ensemble': self.ensemble_dynamics_model.ensemble_model}
            for key, module in modules.items():
                flat_params = U.get_flat_params_from(module).numpy()
                global_min = mpi.mpi_min(np.sum(flat_params))
                global_max = mpi.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

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
                a = self.get_action(x, deterministic=True, certify=False)
                x, reward, terminated, truncated, info = self.eval_env.step(a)
                done = terminated or truncated
                eval_reward += reward
                test_step += 1
            self.logger.store(EvalRet=eval_reward)

    def get_action(self, o, certify=True, deterministic=False):
        if self.in_warm_up:
            if hasattr(self.env, 'safe_controller'):
                a = self.env.safe_controller(o)
            else:
                a = self.env.action_space.sample()
        else:
            th_obs = th.as_tensor(o, dtype=th.float32).to(device=self.device)
            if self.env_sampler.is_init_obs:
                self.controller.reset(obs=th_obs)
            a = self.controller.step(obs=th_obs)
            if certify and self.use_mpsc:
                a = self.mpsc.solve(o, a)
        return a

    def learn(self) -> tuple:
        r"""Main loop: collect experience in env and update/log each epoch."""
        self.run_warmup()
        self.in_warm_up = False

        for self.epoch in range(self.epochs):
            self.estimate_terminal_set(use_polyak=True)
            self.record_video()
            self.epoch_time = time.time()
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
               and not self.in_warm_up \
               and len(self.virtual_buffer) >= self.local_mini_batch_size

    @property
    def is_time_to_update_model_ensemble(self) -> bool:
        return self.local_time_step % self.model_train_freq == 0 \
               and self.real_ratio < 1.0 \
               and len(self.real_buffer) > self.local_mini_batch_size

    def roll_out(self, batch_size: int):
        r"""Rollout >>one<< episode and store to buffer."""
        loggers.debug(f"Do Rollouts...")
        failures, violations = 0, 0
        if self.mpsc is not None and not self.in_warm_up:
            self.mpsc.setup_optimizer()

        for t in range(batch_size):
            o = self.env_sampler.obs.copy()
            a = self.get_action(o)
            next_o, r, done, info = self.env_sampler.step(a)
            self.real_buffer.store(o, a, r, next_o, done)

            if info.get('cost', 0.0) > 0.0:  # Note: use costs insted of not env.observation_space.contains(x) since env_sampler return x=env.reset() and not the terminal state!
                violations += 1
            if self.mpsc is not None and self.mpsc.is_failure:
                failures += 1

            if self.is_time_to_update_policy:
                self.update()
            if self.is_time_to_update_model_ensemble:
                N = mpi.num_procs()
                msg = f'PETS: \tepoch: {self.epoch + 1}/{self.epochs} ' \
                      f'| GLOBAL freq: {self.update_every*N} step: {self.global_time_step}) ' \
                      f'| Local freq: {self.update_every} batch step: {t}/{self.local_batch_size}'
                if not self.in_warm_up and mpi.is_root_process() :
                    print(loggers.colorize(msg, color='cyan'))

                self.train_model_ensemble()
                if self.mpsc is not None and not self.in_warm_up:
                    self.mpsc.setup_optimizer()

            if self.in_warm_up:
                self.global_time_step += 1  # due to broadcasting of init data
            else:
                self.global_time_step += mpi.num_procs()
            self.local_time_step += 1
            o = next_o

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

            from x_mpsc.envs.cartpole import CartPolePlotWrapper
            record_env = CartPolePlotWrapper(record_env)
        else:
            # todo sven: skip TwoLinkArm recording for now..
            return None
            raise ValueError(f"Unknown env: {record_env}")

        print(f"recording with env: {record_env}")

        x, _ = record_env.reset()
        if self.use_mpsc:
            self.mpsc.setup_optimizer()
        pbar = tqdm(range(self.local_mini_batch_size), disable=not mpi.is_root_process())
        nx = self.env.observation_space.shape[0]
        nu = self.env.action_space.shape[0]
        wrapped_models = [
            EnsembleModelCasadiWrapper(
                self.ensemble_dynamics_model,
                model_idx=m) for m in range(self.ensemble_size)
        ]

        for stage in pbar:
            mpi.broadcast(x)
            pbar.set_description(f"Recording video")
            a = self.get_action(x, deterministic=True)
            u_learn = np.clip(a, self.env.action_space.low, self.env.action_space.high)
            if mpi.is_root_process():
                nu = record_env.action_space.shape[0]
                if self.use_mpsc:
                    u_safe = self.mpsc.solve(x, a)
                    Us = self.mpsc.last_Us if self.mpsc.last_Us is not None else np.zeros(
                        (nu, self.mpsc.horizon - 1))
                else:
                    u_safe = a
                    Us = np.zeros((nu, self.mpsc_horizon - 1))
                record_env.plot_current_nominal_trajectory(
                    wrapped_models,
                    self.ensemble_dynamics_model.ensemble_model,
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

    def run_warmup(self):
        loggers.info(f"Run warmup")
        self.roll_out(
            batch_size=self.init_exploration_steps
        )
        self.real_buffer.synchronize()
        self.fit_input_scaler()
        for _ in range(25):
            self.train_model_ensemble()
        self.estimate_terminal_set(False)
        self.logger.save_state(state_dict=self.state_dict, itr=None)
        loggers.info(f"Done with warmup!")

    @property
    def state_dict(self) -> Dict:
        return {'terminal_set': self.terminal_set}

    def train_model_ensemble(self) -> None:
        loggers.debug(f"Train model ensemble...)")
        batch_size = int(min(256//mpi.num_procs(), len(self.real_buffer)))
        state, action, reward, next_state, done = self.real_buffer.return_all()
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = delta_state

        diags = self.ensemble_dynamics_model.train(
            inputs, labels, batch_size=batch_size, holdout_ratio=0.2,
            terminate_early=not self.in_warm_up
        )
        self.logger.store(**diags)

    def update(self):
        num_rollouts = self.update_every * self.policy_train_iterations * mpi.num_procs()
        desc = f"INFO: \tTrain actor-critic"
        with tqdm(total=num_rollouts, desc=desc, ncols=80, disable=not mpi.is_root_process()) as pbar:
            num_real_iters = int(self.real_ratio * self.policy_train_iterations)
            for _ in range(self.update_every * mpi.num_procs()):
                for i in range(self.policy_train_iterations):
                    if self.real_ratio > 0 and i < num_real_iters:
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
    return ProbabilisticEnsemblesTrajectorySampling(
        env_id=env_id,
        **kwargs
    )


# compatible class to OpenAI Baselines learn functions
def learn(env_id, **kwargs) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='pets', env_id=env_id)
    defaults.update(**kwargs)
    alg = ProbabilisticEnsemblesTrajectorySampling(
        env_id,
        **defaults
    )
    ac, env = alg.learn()
    return ac, env

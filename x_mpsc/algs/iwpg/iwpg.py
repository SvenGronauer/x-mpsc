""" Simple PyTorch implementation of Importance-weighted Policy Gradient (IWPG)

Author:     Sven Gronauer (sven.gronauer@tum.de)
based on:   Spinning Up's Repository
            https://spinningup.openai.com/en/latest/

            Successful Ingredients of Policy Gradient Algorithms
            https://github.com/SvenGronauer/successful-ingredients-paper
"""
import numpy as np
import gymnasium as gym
import time
import torch
from copy import deepcopy

# local imports
from x_mpsc.algs import core
import x_mpsc.common.mpi_tools as mpi_tools
import x_mpsc.algs.utils as U
import x_mpsc.common.loggers as loggers


class IWPGAlgorithm(core.OnPolicyGradientAlgorithm):
    def __init__(
            self,
            actor: str,
            ac_kwargs: dict,
            env_id: str,
            epochs: int,
            logger_kwargs: dict,
            adv_estimation_method: str = 'gae',
            alg='iwpg',
            check_freq: int = 25,
            cost_limit: float = 0.0,  # used for constrained MPDs (in child classes)
            entropy_coef: float = 0.01,
            gamma: float = 0.99,
            lam: float = 0.95,  # GAE scalar
            lam_c: float = 0.95,  # GAE scalar for cost estimation
            lambda_lr: float = 0.0,
            max_grad_norm: float = 0.5,
            num_mini_batches: int = 16,  # used for value network training
            optimizer: str = 'Adam',  # policy optimizer
            pi_lr: float = 3e-4,
            steps_per_epoch: int = 32 * 1000,  # number global steps per epoch
            target_kl: float = 0.01,
            train_pi_iterations: int = 80,
            train_v_iterations: int = 5,
            trust_region='plain',  # used for easy filtering in plot utils
            use_cost_value_function: bool = False,
            use_entropy: bool = False,
            use_exploration_noise_anneal: bool = True,
            use_kl_early_stopping: bool = False,
            use_linear_lr_decay: bool = True,
            use_max_grad_norm: bool = False,
            use_reward_scaling: bool = True,
            use_shared_weights: bool = False,
            use_standardized_advantages: bool = False,
            use_standardized_obs: bool = True,
            verbose: bool = True,
            vf_lr: float = 1e-3,
            weight_initialization: str = 'kaiming_uniform',
            save_freq: int = 10,
            seed: int = 0,
            video_freq: int = -1,  # set to positive integer for video recording
            **kwargs  # use to log parameters from child classes
    ):

        # get local parameters before logger instance to avoid unnecessary print
        self.params = locals()

        if isinstance(env_id, str):
            self.env = gym.make(env_id, **kwargs)
        else:
            self.env = env_id
            self.params.pop('env_id')  # already instantiated envs cause errors

        self.adv_estimation_method = adv_estimation_method
        self.alg = alg
        self.check_freq = check_freq
        self.entropy_coef = entropy_coef if use_entropy else 0.0
        self.epoch = 0  # iterated in learn method
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.lam_c = lam_c
        self.local_steps_per_epoch = steps_per_epoch // mpi_tools.num_procs()
        self.logger_kwargs = logger_kwargs
        self.max_grad_norm = max_grad_norm
        self.num_mini_batches = num_mini_batches
        self.pi_lr = pi_lr
        self.save_freq = save_freq
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.target_kl = target_kl
        self.train_pi_iterations = train_pi_iterations
        self.train_v_iterations = train_v_iterations
        self.use_cost_value_function = use_cost_value_function
        self.use_exploration_noise_anneal = use_exploration_noise_anneal
        self.use_kl_early_stopping = use_kl_early_stopping
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_max_grad_norm = use_max_grad_norm
        self.use_reward_scaling = use_reward_scaling
        self.use_standardized_obs = use_standardized_obs
        self.use_standardized_advantages = use_standardized_advantages
        self.video_freq = video_freq
        self.vf_lr = vf_lr

        # ==== Call assertions....
        self._sanity_checks()

        # === Set up logger and save configuration to disk

        self.logger = self._init_logger()
        self.logger.save_config(self.params)
        # save environment settings to disk
        self.logger.save_env_config(env=self.env)
        loggers.set_level(loggers.INFO)

        # === Seeding
        seed += 10000 * mpi_tools.proc_id()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed=seed)

        # === Setup actor-critic module
        self.ac = core.ActorCriticWithCosts(
            actor_type=actor,
            recurrent=False,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            use_standardized_obs=use_standardized_obs,
            use_scaled_rewards=use_reward_scaling,
            use_shared_weights=use_shared_weights,
            weight_initialization=weight_initialization,
            ac_kwargs=ac_kwargs
        )

        # === set up MPI specifics
        self._init_mpi()

        # === Set up experience buffer
        self.buf = core.Buffer(
            actor_critic=self.ac,
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=self.local_steps_per_epoch,
            gamma=gamma,
            lam=lam,
            lam_c=lam_c,
            adv_estimation_method=adv_estimation_method,
            use_scaled_rewards=use_reward_scaling,
            standardize_env_obs=use_standardized_obs,
            standardize_advantages=use_standardized_advantages,
        )

        # Set up optimizers for policy and value function
        self.pi_optimizer = core.get_optimizer(optimizer, module=self.ac.pi,
                                               lr=pi_lr)
        self.vf_optimizer = core.get_optimizer('Adam', module=self.ac.v,
                                               lr=vf_lr)
        if use_cost_value_function:
            self.cf_optimizer = core.get_optimizer('Adam', module=self.ac.c,
                                                   lr=self.vf_lr)
        # setup scheduler for policy learning rate decay
        self.scheduler = self._init_learning_rate_scheduler()

        # Set up model saving
        self.logger.setup_actor_critic_saver(self.ac)
        self.logger.actor_critic_save()

        # setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.total_constraint_violations = 0
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.loss_c_before = 0.0
        self.logger.info('Start with training.')

    def _init_learning_rate_scheduler(self):
        scheduler = None
        if self.use_linear_lr_decay:
            import torch.optim
            def lm(epoch): return 1 - epoch / self.epochs  # linear anneal
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.pi_optimizer,
                lr_lambda=lm
            )
        return scheduler

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
        if mpi_tools.num_procs() > 1:
            loggers.info(f'Started MPI with {mpi_tools.num_procs()} processes.')
            # Avoid slowdowns from PyTorch + MPI combo.
            mpi_tools.setup_torch_for_mpi()
            dt = time.time()
            loggers.info('Sync actor critic parameters...')
            # Sync params across cores: only once necessary, grads are averaged!
            mpi_tools.sync_params(self.ac)
            loggers.info(f'Done! (took {time.time()-dt:0.3f} sec.)')

    def _sanity_checks(self):
        """ Do assertions..."""
        assert self.steps_per_epoch % mpi_tools.num_procs() == 0
        assert self.train_pi_iterations > 0
        assert self.train_v_iterations > 0
        assert isinstance(self.env, gym.Env), 'Env is not the expected type.'

    def algorithm_specific_logs(self):
        """ Use this method to collect log information. """
        pass

    def check_distributed_parameters(self) -> None:
        """Check if parameters are synchronized across all processes."""
        if mpi_tools.num_procs() > 1:
            loggers.info('Check if distributed parameters are synchronous')
            modules = {'Policy': self.ac.pi.net, 'Value': self.ac.v.net}
            for key, module in modules.items():
                flat_params = U.get_flat_params_from(module).numpy()
                global_min = mpi_tools.mpi_min(np.sum(flat_params))
                global_max = mpi_tools.mpi_max(np.sum(flat_params))
                assert np.allclose(global_min, global_max), f'{key} not synced.'

    def compute_loss_c(self, obs, ret) -> torch.Tensor:
        r"""Set up function for computing cost loss."""
        return ((self.ac.c(obs) - ret) ** 2).mean()

    def compute_loss_pi(
            self,
            data: dict,
            lstm_state=None
    ) -> tuple:
        # Policy loss
        dist, _log_p, lstm_state = self.ac.pi(
            data['obs'], data['act'], lstm_state, data['dones'])
        ratio = torch.exp(_log_p - data['log_p'])

        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def compute_loss_v(self, obs, ret) -> torch.Tensor:
        r"""Set up function for computing value loss."""
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def learn(self) -> tuple:
        # Main loop: collect experience in env and update/log each epoch
        for self.epoch in range(self.epochs):
            self.learn_one_epoch()

        # Close opened files to avoid number of open files overflow
        self.logger.close()
        return self.ac, self.env

    def learn_one_epoch(self):
        self.epoch_time = time.time()
        is_last_epoch = self.epoch == self.epochs - 1

        if self.use_exploration_noise_anneal:  # update internals of AC
            self.ac.update(frac=self.epoch / self.epochs)
        self.roll_out()  # collect data and store to buffer
        # Perform policy + value function updates
        # Also updates running statistics
        self.update()
        # Save (or print) information about epoch
        self.log(self.epoch)
        # Check if all models own the same parameter values
        if self.epoch % self.check_freq == 0:
            self.check_distributed_parameters()
        # Save model to disk
        if is_last_epoch or self.epoch % self.save_freq == 0:
            self.logger.save_state(state_dict={}, itr=None)

    def log(self, epoch: int) -> None:
        # Log info about epoch
        total_env_steps = (epoch + 1) * self.steps_per_epoch
        fps = self.steps_per_epoch / (time.time() - self.epoch_time)
        if self.scheduler and self.use_linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()  # step the scheduler if provided
        else:
            current_lr = self.pi_lr

        self.logger.log_tabular('Epoch', epoch + 1)
        self.logger.log_tabular('EpRet', min_and_max=True, std=True)
        self.logger.log_tabular('EpLen', min_and_max=True)
        self.logger.log_tabular('EpCosts', min_and_max=True, std=True)
        self.logger.log_tabular('Values/V', min_and_max=True)
        self.logger.log_tabular('Values/Adv', min_and_max=True)
        if self.use_cost_value_function:
            self.logger.log_tabular('Values/C', min_and_max=True)
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('Loss/DeltaPi')
        self.logger.log_tabular('Loss/DeltaValue')
        if self.use_cost_value_function:
            self.logger.log_tabular('Loss/Cost')
            self.logger.log_tabular('Loss/DeltaCost')
        self.logger.log_tabular('Safety/Violations',
                                self.total_constraint_violations)
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('Misc/StopIter')
        self.logger.log_tabular('Misc/Seed', self.seed)
        self.logger.log_tabular('PolicyRatio')
        self.logger.log_tabular('LR', current_lr)
        if self.use_reward_scaling:
            reward_scale_mean = self.ac.ret_oms.mean.item()
            reward_scale_stddev = self.ac.ret_oms.std.item()
            self.logger.log_tabular('Misc/RewScaleMean', reward_scale_mean)
            self.logger.log_tabular('Misc/RewScaleStddev', reward_scale_stddev)
        if self.use_exploration_noise_anneal:
            noise_std = np.exp(self.ac.pi.log_std[0].item())
            self.logger.log_tabular('Misc/ExplorationNoiseStd', noise_std)
        # some child classes may add information to logs
        self.algorithm_specific_logs()
        self.logger.log_tabular('Misc/TotalEnvSteps', total_env_steps)
        self.logger.log_tabular('Time', int(time.time() - self.start_time))
        self.logger.log_tabular('FPS', int(fps))

        self.logger.dump_tabular()

    def pre_process_data(self, raw_data: dict) -> dict:
        """ Pre-process data, e.g. standardize observations, re-scale rewards if
            enabled by arguments.

        Parameters
        ----------
        raw_data
            dictionary holding information obtain from environment interactions

        Returns
        -------
        dict
            holding pre-processed data, i.e. observations and rewards
        """
        data = deepcopy(raw_data)

        if self.use_standardized_obs:
            assert 'obs' in data
            obs = data['obs']
            data['obs'] = self.ac.obs_oms(obs, clip=False)
        return data

    def roll_out(self) -> None:
        """collect data and store to experience buffer."""
        o, _ = self.env.reset()
        ep_ret, ep_costs, ep_len  = 0., 0., 0
        violations = 0
        done = False
        lstm_state = self.ac.get_initial_lstm_state()

        for t in range(self.local_steps_per_epoch):
            a, v, cv, logp, lstm_state = self.ac.step(
                torch.as_tensor(o, dtype=torch.float32),
                lstm_state,
                torch.as_tensor(done, dtype=torch.float32)
            )
            next_o, r, terminated, timeout, info = self.env.step(a)
            c = info.get('cost', 0.)
            ep_ret += r
            ep_costs += c
            ep_len += 1
            if not self.env.observation_space.contains(next_o):
                violations += 1

            done = terminated or timeout
            self.buf.store(
                obs=o, act=a, rew=r, done=float(done), val=v, logp=logp,
                cost=c, cost_val=cv
            )
            if self.use_cost_value_function:
                self.logger.store(**{
                    'Values/V': v,
                    'Values/C': cv})
            else:
                self.logger.store(**{'Values/V': v})
            
            o = next_o
            batch_is_full = (t == self.local_steps_per_epoch - 1)

            if done or batch_is_full:
                if terminated:
                    v, cv = 0., 0.
                else:  # bootstrap
                    _, v, cv, _, lstm_state = self.ac(
                        torch.as_tensor(o, dtype=torch.float32),
                        lstm_state,
                        torch.as_tensor(done, dtype=torch.float32)
                    )

                self.buf.finish_path(v, cv)
                if done:  
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len,
                                      EpCosts=ep_costs)
                o, _ = self.env.reset()
                ep_ret, ep_costs, ep_len = 0., 0., 0

        self.total_constraint_violations += mpi_tools.mpi_sum(violations)

    def update_running_statistics(self, data) -> None:
        """ Update running statistics, e.g. observation standardization,
        or reward scaling. If MPI is activated: sync across all processes.
        """
        if self.use_standardized_obs:
            self.ac.obs_oms.update(data['obs'])

        if self.use_reward_scaling:
            discounted_ret = data['discounted_ret']

            self.ac.ret_oms.update(data['discounted_ret'].cpu().numpy())

    def update(self) -> None:
        """Update value and policy networks. Note that the order doesn't matter.

        Returns
        -------
            None
        """
        raw_data = self.buf.get()
        # pre-process data: standardize observations, advantage estimation, etc.
        data = self.pre_process_data(raw_data)

        self.update_value_net(data=data)
        self.update_policy_net(data=data)

        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def update_cost_net(self, data: dict) -> None:
        """Some child classes require additional updates,
        e.g. Lagrangian-PPO needs Lagrange multiplier parameter."""
        assert self.use_cost_value_function
        assert hasattr(self, 'cf_optimizer')
        assert 'target_c' in data, f'provided keys: {data.keys()}'
        mbs = self.local_steps_per_epoch // self.num_mini_batches
        assert mbs >= 8, f'Batch size {mbs}<8'
        indices = np.arange(self.local_steps_per_epoch)
        losses = []

        # Train cost value network
        for _ in range(self.train_v_iterations):
            np.random.shuffle(indices)  # shuffle for mini-batch updates
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]

                self.cf_optimizer.zero_grad()
                loss_c = self.compute_loss_c(obs=data['obs'][mb_indices],
                                             ret=data['target_c'][mb_indices])
                loss_c.backward()
                losses.append(loss_c.item())
                # average grads across MPI processes
                mpi_tools.mpi_avg_grads(self.ac.c)
                self.cf_optimizer.step()

        self.logger.store(**{
            'Loss/DeltaCost': np.mean(losses) - self.loss_c_before,
            'Loss/Cost': self.loss_c_before,
        })

    def update_policy_net(self, data) -> None:
        # Get loss and info values before update
        initial_lstm_state = self.ac.get_initial_lstm_state()
        pi_l_old, pi_info_old = self.compute_loss_pi(data, initial_lstm_state)
        self.loss_pi_before = pi_l_old.item()
        if self.use_cost_value_function:
            self.loss_c_before = self.compute_loss_c(data['obs'], data['target_c']).item()
        # get prob. distribution before updates: used to measure KL distance
        p_dist, _ = self.ac.pi.dist(data['obs'], initial_lstm_state, data['dones'])

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iterations):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data, initial_lstm_state)
            loss_pi.backward()
            if self.use_max_grad_norm:  # apply L2 norm
                torch.nn.utils.clip_grad_norm_(
                    self.ac.pi.parameters(),
                    self.max_grad_norm)
            # average grads across MPI processes
            mpi_tools.mpi_avg_grads(self.ac.pi)
            self.pi_optimizer.step()
            q_dist, _ = self.ac.pi.dist(data['obs'],  initial_lstm_state, data['dones'])
            torch_kl = torch.distributions.kl.kl_divergence(
                p_dist, q_dist).mean().item()
            if self.use_kl_early_stopping:
                # average KL for consistent early stopping across processes
                if mpi_tools.mpi_avg(torch_kl) > self.target_kl:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break

        # track when policy iteration is stopped; Log changes from update
        self.logger.store(**{
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/StopIter': i + 1,
            'Values/Adv': data['adv'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': torch_kl,
            'PolicyRatio': pi_info['ratio']
        })

    def update_value_net(self, data: dict) -> None:
        mbs = int(self.local_steps_per_epoch // self.num_mini_batches)
        assert mbs > 1, f'Batch size must be greater than 1. Currently: {mbs}'
        if mbs < 8:
            loggers.warn(f'Batch size is low: {mbs}')

        loss_v = self.compute_loss_v(data['obs'], data['target_v'])
        self.loss_v_before = loss_v.item()

        indices = np.arange(self.local_steps_per_epoch)
        val_losses = []
        for _ in range(self.train_v_iterations):
            np.random.shuffle(indices)  # shuffle for mini-batch updates
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(
                    obs=data['obs'][mb_indices],
                    ret=data['target_v'][mb_indices])
                loss_v.backward()
                val_losses.append(loss_v.item())
                # average grads across MPI processes
                mpi_tools.mpi_avg_grads(self.ac.v)
                self.vf_optimizer.step()

        self.logger.store(**{
            'Loss/DeltaValue': np.mean(val_losses) - self.loss_v_before,
            'Loss/Value': self.loss_v_before,
        })


def get_alg(env_id, **kwargs) -> core.Algorithm:
    return IWPGAlgorithm(
        env_id=env_id,
        **kwargs
    )


# compatible class to OpenAI Baselines learn functions
def learn(env_id, **kwargs) -> tuple:
    defaults = U.get_defaults_kwargs(alg='iwpg', env_id=env_id)
    defaults.update(**kwargs)
    alg = IWPGAlgorithm(
        env_id=env_id,
        **defaults
    )
    ac, env = alg.learn()
    return ac, env

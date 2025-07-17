from __future__ import annotations
import numpy as np
# from x_mpsc.models import DynamicsModel


class PredictEnv:
    def __init__(
            self,
            model, #: DynamicsModel,
            env_name: str
    ):
        self.dynamics_model = model
        self.env_name = env_name

    def _reward_fn(self, obs, act, next_obs):
        if self.env_name == "SimplePendulum-v0":
            if obs.ndim == 2 and act.ndim == 2:
                r = np.cos(obs[:, 0]) - 0.001 * obs[:, 1] ** 2 - 0.001 * (act[:, 0]**2)
            else:
                r = np.cos(obs[0]) - 0.001 * obs[1] ** 2 - 0.001 * (act**2)
            return r
        elif self.env_name == "TwoLinkArm-v0":
            if obs.ndim == 2 and act.ndim == 2:
                dones = self._termination_fn(next_obs)
                r = -(np.linalg.norm(obs[:, 2:4], axis=-1) + 1e-3 * np.linalg.norm(act, axis=-1))
                r -= 100 * dones.astype(float)
            else:
                done = self._termination_fn(next_obs.reshape((1, -1)))
                r = -(np.linalg.norm(obs[2:4]) + 1e-3 * np.linalg.norm(act))
                if done.all():
                    r -= 100
            return r
        elif self.env_name == "SafeCartPole-v0":
            if obs.ndim == 2 and act.ndim == 2:
                dones = self._termination_fn(next_obs)
                r = -(np.linalg.norm(obs, axis=-1)**2 + 1e-4 * np.linalg.norm(act, axis=-1)**2)
                r -= 100 * dones.astype(float)
            else:
                r = -(np.linalg.norm(obs)**2 + 1e-4 * np.linalg.norm(act)**2)
                done = self._termination_fn(next_obs.reshape((1, -1)))
                if done.all():
                    r -= 100
            return r
        elif self.env_name == "SafeDrone-v0":
            if obs.ndim == 2 and act.ndim == 2:
                dones = self._termination_fn(next_obs)
                deltas = np.array([0, 0, 1]) - obs[:, :3]
                r = -1.0 * np.linalg.norm(deltas, axis=-1)**2
                r -= 100 * dones.astype(float)
                r += 1.0 * (~dones).astype(float)
            else:
                delta = np.array([0, 0, 1]) - obs[:3]
                r = -1.0 * np.linalg.norm(delta)**2
                done = self._termination_fn(next_obs.reshape((1, -1)))
                r += -100 if done.all() else 1.0
            return r
        elif self.env_name == "Ant-v4":
            """
                | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Unit                     |
                |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
                | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | torso                                  | free  | position (m)             |
                | 1   | x-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
                | 2   | y-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
                | 3   | z-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
                | 4   | w-orientation of the torso (centre)                          | -Inf   | Inf    | torso                                  | free  | angle (rad)              |
                | 5   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
                | 6   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
                | 7   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
                | 8   | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
                | 9   | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
                | 10  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
                | 11  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
                | 12  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
                | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
                | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
                | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | torso                                  | free  | velocity (m/s)           |
                | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
                | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular velocity (rad/s) |
                | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | torso                                  | free  | angular ve
            
                reward function; see:
                https://github.com/rodrigodelazcano/gym/blob/9e66399b4ef04c1534c003641802e2ac1363e8a6/gym/envs/mujoco/ant_v4.py#L303
            """
            if obs.ndim == 2 and act.ndim == 2:
                x_velocity = obs[:, 13]
                forward_reward = x_velocity
                healthy_reward = np.ones(obs.shape[0])
                costs = 0.5 * np.sum(np.square(act), axis=-1)
                r = forward_reward + healthy_reward - costs
            else:
                x_velocity = obs[13]
                forward_reward = x_velocity
                healthy_reward = 1
                costs = 0.5 * np.sum(np.square(act))
                r = forward_reward + healthy_reward - costs
            return r

        elif self.env_name == "Hopper-v4" or self.env_name == "SafeHopper-v4":
            healthy_reward = 1.0
            if obs.ndim == 2 and act.ndim == 2:
                costs = 1e-3 * np.sum(np.square(act), axis=-1)
                forward_reward = obs[:, 5]  # note: this only an approximation to the true x-velocity
                r = forward_reward + healthy_reward - costs
            else:
                costs = 1e-3 * np.sum(np.square(act))
                forward_reward = obs[5]  # note: this only an approximation to the true x-velocity
                r = forward_reward + healthy_reward - costs
            return r
        else:
            raise ValueError(f"reward func not implemented for: {self.env_name}")

    def _termination_fn(self, next_obs):
        if self.env_name == "SimplePendulum-v0":
            assert next_obs.ndim == 2
            angle = np.logical_or(next_obs[:, 0] > 2.0833*np.pi,
                                  next_obs[:, 0] < 0.25*np.pi)
            velocity = np.logical_or(next_obs[:, 1] > 8.0,
                                     next_obs[:, 1] < -8.0)
            return np.logical_or(angle, velocity)
        elif self.env_name == "TwoLinkArm-v0":
            assert next_obs.ndim == 2
            done = np.logical_or(np.abs(next_obs[:, 0]) > 0.5,
                                 np.abs(next_obs[:, 1]) > 0.5)
            return done
        elif self.env_name == "SafeCartPole-v0":
            assert next_obs.ndim == 2
            done = np.logical_or(np.abs(next_obs[:, 0]) > 2.4,
                                 np.abs(next_obs[:, 2]) > 12 * np.pi/180)
            return done
        elif self.env_name == "SafeDrone-v0":
            assert next_obs.ndim == 2
            done = np.logical_or(np.abs(next_obs[:, 3]) > 10,
                                 np.abs(next_obs[:, 4]) > 10)
            return done
        elif self.env_name == "Ant-v4":
            assert next_obs.ndim == 2
            is_fin = np.isfinite(next_obs).all(axis=-1)
            is_in_height = np.logical_and(0.2 <= next_obs[:, 0],  next_obs[:, 0]<= 1.0)
            not_done = np.logical_and(is_fin, is_in_height)

            done = ~not_done
            return done
        elif self.env_name == "Hopper-v4" or self.env_name == "SafeHopper-v4":
            # ### Episode Termination
            #     The hopper is said to be unhealthy if any of the following happens:
            #     1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, else `observation[2:]`) is no longer contained in the closed interval specified by the argument `healthy_state_range`
            #     2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, else `observation[1]`) is no longer contained in the closed interval specified by the argument `healthy_z_range` (usually meaning that it has fallen)
            #     3. The angle (`observation[1]` if  `exclude_current_positions_from_observation=True`, else `observation[2]`) is no longer contained in the closed interval specified by the argument `healthy_angle_range`
            assert next_obs.ndim == 2

            min_state, max_state = (-100.0, 100.0)
            min_z, max_z = (0.7, float("inf"))
            min_angle, max_angle = (-0.2, 0.2)

            healthy_state = np.logical_and(min_state < next_obs[:, 1:],
                                           next_obs[:, 1:] < max_state).all(axis=-1)
            healthy_z = np.logical_and(min_z < next_obs[:, 0],  next_obs[:, 0]< max_z)
            healthy_angle = np.logical_and(min_angle < next_obs[:, 1],
                                           next_obs[:, 1] < max_angle)
            not_done = np.logical_and(healthy_state, healthy_z)
            not_done = np.logical_and(healthy_angle, not_done)

            # speed limit:
            not_done = np.logical_and(np.abs(next_obs[:, 5]) < 2.5, not_done)

            done = ~not_done
            return done
        else:
            raise ValueError(f"termination func not implemented for: {self.env_name}")

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ ensemble_size, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances+1e-10).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False, model_idx: int = -1):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.dynamics_model.predict(inputs)

        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape

        idxs = [model_idx, ] if model_idx >= 0 else self.dynamics_model.elite_model_idxes

        model_idxes = np.random.choice(idxs, size=batch_size)

        batch_idxes = np.arange(0, batch_size)
        nn_preds = ensemble_samples[model_idxes, batch_idxes]
        # model_means = ensemble_model_means[model_idxes]
        # model_stds = ensemble_model_stds[model_idxes]
        if self.dynamics_model.prior_dynamics_model is not None:
            f = self.dynamics_model.prior_dynamics_model.predict_next_state
            if obs.ndim == 1:
                next_obs = f(obs, act).reshape(obs.shape) + nn_preds
            else:
                next_obs = np.array(list(map(f, obs, act))).reshape(obs.shape) + nn_preds
            # print(next_obs)
        else:
            next_obs = nn_preds
        rewards = self._reward_fn(obs, act, next_obs)
        dones = self._termination_fn(next_obs)
        # log_prob, dev = self._get_logprob(next_obs, ensemble_model_means, ensemble_model_vars)

        if return_single:
            next_obs = next_obs[0]
            rewards = rewards[0]
            dones = dones[0]

        # info = {'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, dones, {}

import gymnasium as gym
import numpy as np


class EnvSampler:
    def __init__(self,
                 env: gym.Env,
                 ):
        self.env = env
        self.obs, _ = self.env.reset()
        self.is_init_obs = True

        self.N = 2

        self.ep_ret = 0.
        self.ep_len = 0
        self.last_episode_returns = []
        self.last_episode_lengths = []
        self.random_policy_reward = self.get_random_policy_reward()

    def step(self, a: np.ndarray):
        next_o, r, terminated, truncated, info = self.env.step(a)
        done = terminated or truncated
        self.ep_ret += r
        self.ep_len += 1
        if done:
            self.last_episode_returns.append(self.ep_ret)
            self.last_episode_lengths.append(self.ep_len)
            self.obs, _ = self.env.reset()
            self.ep_ret, self.ep_len = 0., 0
            self.is_init_obs = True
        else:
            self.obs = next_o
            self.is_init_obs = False
        return next_o, r, done, info

    @property
    def average_trajectory_returns(self):
        if len(self.last_episode_returns) == 0:
            return self.random_policy_reward
        elif len(self.last_episode_returns) >= self.N:
            return np.mean(self.last_episode_returns[-self.N:])
        else:
            return np.mean(self.last_episode_returns)
        
    @property
    def average_trajectory_lengths(self):
        if len(self.last_episode_lengths) == 0:
            return 0.0
        elif len(self.last_episode_lengths) >= self.N:
            return np.mean(self.last_episode_lengths[-self.N:])
        else:
            return np.mean(self.last_episode_lengths)

    def get_random_policy_reward(self, num_episodes=100):
        rets = []
        for _ in range(num_episodes):
            terminal = False
            self.env.reset()
            while not terminal:
                a = self.env.action_space.sample()
                x, r, terminated, truncated, info = self.env.step(a)
                done = terminated or truncated
                self.ep_ret += r
                self.ep_len += 1
                if done:
                    rets.append(self.ep_ret)
                    self.obs, _ = self.env.reset()
                    self.ep_ret, self.ep_len = 0., 0
        return np.mean(rets)

import numpy as np
import torch
import x_mpsc.algs.core as core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim),
                                 dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.act_learn_buf = np.zeros(core.combined_shape(size, act_dim),
                                      dtype=np.float32)
        self.act_unsafe_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def __len__(self):
        return self.size

    def get_all_safe_samples(self):
        safe_samples = []
        for i in range(self.size):
            if self.act_unsafe_buf[i] < 0.5:  # action is safe
                safe_samples.append(self.obs_buf[i])
        return np.asarray(safe_samples, dtype=np.float32)

    def return_all(self, as_numpy: bool = False):
        return self.sample_batch(self.size, as_numpy=as_numpy)

    def store(self, obs, act, rew, next_obs, done, act_learn, is_action_safe):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.act_learn_buf[self.ptr] = act_learn
        self.act_unsafe_buf[self.ptr] = not is_action_safe
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, as_numpy: bool = False):
        batch_size = int(min(batch_size, self.size))
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     act_learn=self.act_learn_buf[idxs],
                     act_is_unsafe=self.act_unsafe_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        if as_numpy:
            return batch
        else:
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in
                    batch.items()}

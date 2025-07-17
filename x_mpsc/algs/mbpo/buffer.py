import numpy as np
import torch as th

# local imports
import x_mpsc.common.mpi_tools as mpi
import x_mpsc.algs.core as core
from x_mpsc.algs.utils import get_device


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim),
                                 dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.feasible_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = get_device()

    def __len__(self):
        return self.size

    def synchronize(self):
        r"""Broadcast replay buffer of root process to all other processes."""
        mpi.broadcast(self.obs_buf)
        mpi.broadcast(self.obs2_buf)
        mpi.broadcast(self.act_buf)
        mpi.broadcast(self.rew_buf)
        mpi.broadcast(self.done_buf)

        global_min = mpi.mpi_min(self.ptr)
        global_max = mpi.mpi_max(self.ptr)
        assert np.allclose(global_min, global_max), f'Buffer pointer not synced'

    def push_batch(self, obs, acs, rews, next_obs, dones):
        assert obs.shape[0] == acs.shape[0] == dones.shape[0]
        assert rews.shape[0] == next_obs.shape[0]
        batch_size = obs.shape[0]

        if (self.ptr + batch_size) <= self.max_size:
            _slice = slice(self.ptr, self.ptr + batch_size)
            self.obs_buf[_slice] = obs
            self.act_buf[_slice] = acs
            self.rew_buf[_slice] = rews
            self.obs2_buf[_slice] = next_obs
            self.done_buf[_slice] = dones
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

    def store(self, obs, act, rew, next_obs, done, feasible: bool):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.feasible_buf[self.ptr] = feasible
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = int(min(self.size + 1, self.max_size))

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: th.as_tensor(v, dtype=th.float32).to(device=self.device)
                for k, v in batch.items()}

    def return_all(self):
        return self.obs_buf[:self.size], self.act_buf[:self.size], \
               self.rew_buf[:self.size], self.obs2_buf[:self.size], \
               self.done_buf[:self.size]

    def return_all_obs_where_mpsc_was_feasible(self):
        idxs = self.feasible_buf.astype(bool)
        return self.obs_buf[idxs]

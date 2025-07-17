import numpy as np
import torch

# local imports
import x_mpsc.algs.core as core
import x_mpsc.common.mpi_tools as mpi


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
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.con_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def __len__(self):
        return self.size

    def return_all(self, as_numpy: bool = False):
        return self.sample_batch(self.size, as_numpy=as_numpy)

    def store(self, obs, act, rew, next_obs, done, con):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.con_buf[self.ptr] = con
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, as_numpy: bool = False):
        batch_size = int(min(batch_size, self.size))
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     con=self.con_buf[idxs],
                     done=self.done_buf[idxs])
        if as_numpy:
            return batch
        else:
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in
                    batch.items()}

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

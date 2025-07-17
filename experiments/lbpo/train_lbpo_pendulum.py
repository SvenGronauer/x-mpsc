import gymnasium as gym
import os 

from x_mpsc.algs.lbpo.LBPO import LBPO
from x_mpsc.algs.lbpo import core
from x_mpsc.algs.lbpo.ppo_utils.mpi_tools import mpi_fork
from x_mpsc.algs.lbpo.ppo_utils.run_utils import setup_logger_kwargs

# training parameters
exp_name = 'LBPOSimplePendulum'
env_name = 'SimplePendulum-v0'
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
n_cpus = 8
seeds = [0, 15, 25]
steps_epoch = 8000
n_epochs = 1000
target_l2 = 0.012
cost_limit = 0.0

# model parameters
hid = [256, 256]  # actor critic hidden layers
gamma = 0.99
beta = 0.01
beta_thres = 0.05

mpi_fork(n_cpus)  # run parallel code with mpi

for seed in seeds:

    logger_kwargs = setup_logger_kwargs(exp_name, seed=seed, data_dir=data_dir)

    LBPO(lambda : gym.make(env_name), env_name= env_name, actor_critic=core.MLPActorCriticTD3trust,
        ac_kwargs=dict(hidden_sizes=hid), gamma=gamma, seed=seed, steps_per_epoch=steps_epoch, 
        epochs=n_epochs, target_l2=target_l2, cost_lim=cost_limit, beta=beta, beta_thres = beta_thres,
        logger_kwargs=logger_kwargs)

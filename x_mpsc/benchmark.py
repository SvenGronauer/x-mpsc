""" Benchmark different algorithms and setups on environments.

Author: Sven Gronauer
Date:   28.11.2022

-----
Usage

    alg_setup = {
        'trpo2': {"target_kl": [0.01, 0.02, 0.03]},
        'cpo': {'target_kl': [0.01, 0.02, 0.03], 'lam_c': [1.0, 0.9, 0.5]}
    }
    bench = Benchmark(
        alg_setup,
        env_id='SafetyHopperRun-v0',
        log_dir=args.log_dir,
        num_cores=args.num_cores,
        num_runs=args.num_runs,
    )
    bench.run()
"""
import sys
import os
import warnings
import json
from itertools import product
import torch as th
import gymnasium as gym
import psutil
import atexit
import numpy as np

# local imports
import x_mpsc.common.mpi_tools as mpi
import x_mpsc.common.loggers as loggers
from x_mpsc.common import utils



class EnvironmentEvaluator(object):
    def __init__(self, log_dir, log_costs=True):

        self.log_dir = log_dir
        self.env = None
        self.ac = None
        self.log_costs = log_costs

        # open returns.csv file at the beginning to avoid disk access errors
        # on our HPC servers...
        if mpi.proc_id() == 0:
            os.makedirs(log_dir, exist_ok=True)
            self.ret_file_name = 'returns.csv'
            self.ret_file = open(os.path.join(log_dir, self.ret_file_name), 'w')
            # Register close function is executed for normal program termination
            atexit.register(self.ret_file.close)
            if log_costs:
                self.c_file_name = 'costs.csv'
                self.costs_file = open(os.path.join(log_dir, self.c_file_name), 'w')
                atexit.register(self.costs_file.close)
        else:
            self.ret_file_name = None
            self.ret_file = None
            if log_costs:
                self.c_file_name = None
                self.costs_file = None

    def close(self):
        """Close opened output files immediately after training in order to
        avoid number of open files overflow. Avoids the following error:
        OSError: [Errno 24] Too many open files
        """
        if mpi.proc_id() == 0:
            self.ret_file.close()
            if self.log_costs:
                self.costs_file.close()

    def eval(self, env, ac, num_evaluations):
        """ Evaluate actor critic module for given number of evaluations.
        """
        self.ac = ac
        self.ac.eval()  # disable exploration noise

        if isinstance(env, gym.Env):
            self.env = env
        elif isinstance(env, str):
            self.env = gym.make(env)
        else:
            raise TypeError('Env is not of type: str, gym.Env')

        size = mpi.num_procs()
        num_local_evaluations = num_evaluations // size
        returns = np.zeros(num_local_evaluations, dtype=np.float32)
        costs = np.zeros(num_local_evaluations, dtype=np.float32)
        ep_lengths = np.zeros(num_local_evaluations, dtype=np.float32)

        for i in range(num_local_evaluations):
            returns[i], ep_lengths[i], costs[i] = self.eval_once()
        # Gather returns from all processes
        # Note: only root process owns valid data...
        returns = list(mpi.gather_and_stack(returns))
        costs = list(mpi.gather_and_stack(costs))

        # now write returns as column into output file...
        if mpi.proc_id() == 0:
            self.write_to_file(self.ret_file, contents=returns)
            print('Saved to:', os.path.join(self.log_dir, self.ret_file_name))
            if self.log_costs:
                self.write_to_file(self.costs_file, contents=costs)
            print(f'Mean Ret: { np.mean(returns)} \t'
                  f'Mean EpLen: {np.mean(ep_lengths)} \t'
                  f'Mean Costs: {np.mean(costs)}')

        self.ac.train()  # back to train mode
        return np.array(returns), np.array(ep_lengths), np.array(costs)

    def eval_once(self):
        assert not self.ac.training, 'Call actor_critic.eval() beforehand.'
        done = False
        x, _ = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0

        while not done:
            obs = th.as_tensor(x, dtype=th.float32)
            action, value, *_ = self.ac(obs)
            x, r, done, info = self.env.step(action)
            ret += r
            costs += info.get('cost', 0.)
            episode_length += 1

        return ret, episode_length, costs

    @staticmethod
    def write_to_file(file, contents: list):
        if mpi.proc_id() == 0:
            column = [str(x) for x in contents]
            file.write("\n".join(column) + "\n")
            file.flush()


def run_training(skip_eval=False, **kwargs) -> None:
    alg = kwargs.pop('alg')
    env_id = kwargs.pop('env_id')
    logger_kwargs = kwargs.pop('logger_kwargs')
    evaluator = EnvironmentEvaluator(
        log_dir=logger_kwargs['log_dir'],
        log_costs=True)
    learn = utils.get_learn_function(alg)
    ac, env = learn(env_id,
                    logger_kwargs=logger_kwargs,
                    **kwargs)
    if not skip_eval:
        evaluator.eval(
            ac=ac,
            env=env,
            num_evaluations=128)
    # close output files after evaluation to limit number of open files
    evaluator.close()


class Benchmark:
    """ Benchmark several algorithms on certain environments.

        important input paramater:

        alg_setup: dict
            {
                'trpo': {"target_kl": [0.01, 0.001], "gamma": [0.95, 0.9]}
            }
    """

    def __init__(self,
                 alg_setup: dict,
                 env_ids: list,
                 log_dir: str,
                 num_cores: int,
                 num_runs: int,
                 env_specific_kwargs: dict,
                 skip_eval: bool = False,
                 use_mpi: bool = True,
                 init_seed: int = 0
                 ) -> None:
        self.env_ids = env_ids
        self.env_specific_kwargs = env_specific_kwargs
        self.log_dir = log_dir
        self.alg_setup = alg_setup
        self.init_seed = init_seed
        self.num_cores = num_cores
        self.num_runs = num_runs
        self.skip_eval = skip_eval
        self.use_mpi = use_mpi
        # Exclude hyper-threading and round cores to anything in: [2, 4, 8, etc]
        physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))

        # Use number of physical cores as default. If also hardware
        # threading CPUs should be used, enable this by:
        use_num_of_threads = True if num_cores > physical_cores else False
        if mpi.mpi_fork(num_cores, use_number_of_threads=use_num_of_threads):
            sys.exit()

    @classmethod
    def _convert_to_dict(cls, param_grid) -> dict:
        # convert string to dict
        if isinstance(param_grid, str):
            param_grid = json.loads(param_grid)
        elif isinstance(param_grid, dict):
            pass
        else:
            raise TypeError(f'param_grid of type: {type(param_grid)}')
        return param_grid

    def run(self):
        """Run parameter grid over all MPI processes. No scheduling required."""
        init_seed = self.init_seed
        for env_id in self.env_ids:
            for i in range(self.num_runs):
                for alg_name, param_grid in self.alg_setup.items():
                    param_grid = self._convert_to_dict(param_grid)
                    exp_name = os.path.join(env_id, alg_name)

                    for param_set in product(*param_grid.values()):
                        grid_kwargs = dict(zip(param_grid.keys(), param_set))

                        if mpi.is_root_process():
                            msg = f'Run #{i} (with seed={init_seed}) and kwargs:\n{grid_kwargs}'
                            msg = loggers.colorize(msg, color='yellow', bold=True)
                            print(msg)

                        kwargs = utils.get_defaults_kwargs(
                            alg=alg_name,
                            env_id=env_id
                        )
                        logger_kwargs = loggers.setup_logger_kwargs(
                            base_dir=self.log_dir,
                            exp_name=exp_name,
                            seed=init_seed,
                            level=0,
                            use_tensor_board=True,
                            verbose=False)
                        kwargs.update(logger_kwargs=logger_kwargs,
                                      seed=init_seed,
                                      alg=alg_name,
                                      env_id=env_id)
                        # firstly, update environment specifics
                        kwargs.update(**self.env_specific_kwargs[env_id])
                        # secondly, pass the grid search parameters...
                        kwargs.update(**grid_kwargs)
                        run_training(skip_eval=self.skip_eval, **kwargs)
                        init_seed += 1

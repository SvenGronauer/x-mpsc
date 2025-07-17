import argparse
import numpy as np
import psutil
import sys
import time
import warnings
import getpass
from typing import Optional, Tuple
import torch
import os
import gymnasium as gym

# local imports
import x_mpsc.envs  # noqa
import x_mpsc.common.loggers as loggers
from x_mpsc.common.mpi_tools import mpi_fork, is_root_process, mpi_print, USE_MPI
from x_mpsc.common.loggers import setup_logger_kwargs
from x_mpsc.common import utils


class Model(object):
    r""" Introduce an API which is similar to keras to train RL algorithms."""

    def __init__(self,
                 alg: str,
                 env_id: str,
                 log_dir: str,
                 init_seed: int,
                 algorithm_kwargs: dict = {},
                 use_mpi: bool = False,
                 ) -> None:
        """ Class Constructor  """
        self.alg = alg
        self.env_id = env_id
        self.log_dir = log_dir
        self.init_seed = init_seed
        self.num_runs = 1
        self.num_cores = 1  # set by compile()-method
        self.training = False
        self.compiled = False
        self.trained = False
        self.use_mpi = use_mpi

        self.default_kwargs = utils.get_defaults_kwargs(alg=alg,
                                                        env_id=env_id)
        self.kwargs = self.default_kwargs.copy()
        self.kwargs['seed'] = init_seed
        self.kwargs.update(**algorithm_kwargs)
        self.logger_kwargs = None  # defined by compile (a specific seed might be passed)
        self.env_alg_path = os.path.join(self.env_id, self.alg)

        # assigned by class methods
        self.model = None
        self.env = None
        self.scheduler = None


    def _evaluate_model(self) -> None:
        pass
        # raise NotImplementedError
        # from x_mpsc.common.experiment_analysis import EnvironmentEvaluator
        # evaluator = EnvironmentEvaluator(log_dir=self.logger_kwargs['log_dir'])
        # evaluator.eval(env=self.env, ac=self.model, num_evaluations=128)
        # # Close opened files to avoid number of open files overflow
        # evaluator.close()

    def compile(self,
                num_cores=os.cpu_count(),
                exp_name: Optional[str] = None,
                **kwargs_update
                ) -> object:
        """Compile the model.

        Either use mpi for parallel computation or run N individual processes.

        Parameters
        ----------
        num_cores
        exp_name
        kwargs_update

        Returns
        -------

        """
        self.kwargs.update(kwargs_update)
        _seed = self.kwargs.get('seed', self.init_seed)

        if exp_name is not None:
            exp_name = os.path.join(self.env_alg_path, exp_name)
        else:
            exp_name = self.env_alg_path
        self.logger_kwargs = setup_logger_kwargs(base_dir=self.log_dir,
                                                 exp_name=exp_name,
                                                 seed=_seed)
        self.compiled = True
        self.num_cores = num_cores
        return self

    def _eval_once(self, actor_critic, env, render) -> tuple:
        done = False
        self.env.render() if render else None
        x, _ = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            self.env.render() if render else None
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, info = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0)
            ret += r
            episode_length += 1
        return ret, episode_length, costs

    def eval(self, **kwargs) -> None:

        self.model.eval()  # Set in evaluation mode before evaluation
        self._evaluate_model()
        self.model.train()  # switch back to train mode

    def fit(self, epochs=None, env=None) -> None:
        """ Train the model for a given number of epochs.

        Parameters
        ----------
        epochs: int
            Number of epoch to train. If None, use the standard setting from the
            defaults.py of the corresponding algorithm.
        env: gym.Env
            provide a custom environment for fitting the model, e.g. pass a
            virtual environment (based on NN approximation)

        Returns
        -------
        None

        """
        assert self.compiled, 'Call model.compile() before model.fit()'

        # single model training
        if epochs is None:
            epochs = self.kwargs.pop('epochs')
        else:
            self.kwargs.pop('epochs')  # pop to avoid double kwargs

        # fit() can also take a custom env, e.g. a virtual environment
        env_id = self.env_id if env is None else env

        learn_func = utils.get_learn_function(self.alg)
        ac, env = learn_func(
            env_id=env_id,
            logger_kwargs=self.logger_kwargs,
            epochs=epochs,
            **self.kwargs
        )
        self.model = ac
        self.env = env
        self.trained = True

    def play(self) -> None:
        """ Visualize model after training."""
        assert self.trained, 'Call model.fit() before model.play()'
        self.eval(episodes=5, render=True)

    def summary(self):
        """ print nice outputs to console."""
        raise NotImplementedError

def get_training_command_line_args(alg: Optional[str] = None,
                                   env: Optional[str] = None,
                                   ) -> Tuple[list, list]:
    r"""Fetches command line arguments from sys.argv.

    Parameters
    ----------
    alg: over-writes console
    env

    Returns
    -------
    Tuple of two lists
    """

    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    if USE_MPI:
        physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    else:
        physical_cores = 1
    # Seed must be < 2**32 => use 2**16 to allow seed += 10000*proc_id() for MPI
    random_seed = int(time.time()) % 2**16
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Algorithm argument is set to passed argument `alg`
    if alg is not None:
        parser.add_argument('--alg', type=str, default=alg)
    else:  # --add alg as required console argument
        parser.add_argument(
            '--alg', type=str, required=True,
            help='Choose from: {iwpg, ppo, trpo, npg}')

    parser.add_argument(
        '--cores', '-c', type=int, default=physical_cores,
        help=f'Number of cores used for calculations.')
    parser.add_argument(
        '--debug', action='store_true',
        help='Show debug prints during training.')

    # Environment argument is set to passed argument `env`
    if env is not None:
        parser.add_argument('--env', type=str, default=env)
    else:
        parser.add_argument(
            '--env', type=str, required=True,
            help='Example: HopperBulletEnv-v0')

    parser.add_argument(
        '--no-mpi', action='store_true',
        help='Do not use MPI for parallel execution.')
    parser.add_argument(
        '--pi', nargs='+',  # creates args as list: pi=['64,', '64,', 'relu']
        help='Structure of policy network. Usage: --pi 64 64 relu')
    parser.add_argument(
        '--play', action='store_true',
        help='Visualize agent after training.')
    parser.add_argument(
        '--seed', default=random_seed, type=int,
        help=f'Define the init seed, e.g. {random_seed}')
    parser.add_argument(
        '--search', action='store_true',
        help='If given search over learning rates.')

    user_name = getpass.getuser()
    parser.add_argument(
        '--log-dir', type=str, default=os.path.join('/var/tmp/', user_name),
        help='Define a custom directory for logging.')

    args, unparsed_args = parser.parse_known_args()

    return args, unparsed_args


def run_training(args, unparsed_args, exp_name=None):
    r"""Executes one training loop with given parameters."""

    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    if USE_MPI:
        physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    else:
        physical_cores = 1

    # Use number of physical cores as default. If also hardware threading CPUs
    # should be used, enable this by the use_number_of_threads=True
    use_number_of_threads = True if args.cores > physical_cores else False
    if mpi_fork(args.cores, use_number_of_threads=use_number_of_threads):
        # Re-launches the current script with workers linked by MPI
        sys.exit()
    mpi_print('Unknowns:', unparsed_args)

    # update algorithm kwargs with unparsed arguments from command line
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [eval(v) for v in unparsed_args[1::2]]
    unparsed_kwargs = {k: v for k, v in zip(keys, values)}

    algorithm_kwargs = utils.get_defaults_kwargs(alg=args.alg, env_id=args.env)

    # update algorithm_kwargs with unparsed arguments from command line:
    algorithm_kwargs.update(**unparsed_kwargs)

    if args.pi is not None:
        hidden_sizes = tuple(eval(s) for s in args.pi[:-1])
        assert np.all([isinstance(s, int) for s in hidden_sizes]), \
            f'Hidden sizes must be of type: int'
        activation = args.pi[-1]
        assert isinstance(activation, str), 'Activation expected as string.'

        algorithm_kwargs['ac_kwargs']['pi']['hidden_sizes'] = hidden_sizes
        algorithm_kwargs['ac_kwargs']['pi']['activation'] = activation

    mpi_print('=' * 55)
    mpi_print('Parsed algorithm kwargs:')
    mpi_print(algorithm_kwargs)
    mpi_print('='*55)

    model = Model(
        alg=args.alg,
        env_id=args.env,
        log_dir=args.log_dir,
        init_seed=args.seed,
        algorithm_kwargs=algorithm_kwargs,
        use_mpi=not args.no_mpi
    )
    model.compile(num_cores=args.cores, exp_name=exp_name)

    model.fit()
    model.eval()
    if args.play:
        model.play()


if __name__ == '__main__':
    args, unparsed_args = get_training_command_line_args()
    run_training(args, unparsed_args)

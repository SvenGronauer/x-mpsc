import argparse
import numpy as np
import psutil

import x_mpsc  # noqa
from x_mpsc.benchmark import Benchmark
import x_mpsc.common.loggers as loggers


alg_setup = {
    'mbpo': {
        'use_prior_model': [False, True, ],
        'ensemble_size': [5, ],
        'delay_factor': [1, 5],
        'mpsc_horizon': [5, 7, ],
        'use_mpsc': [True, ]
    },
}

env_specific_kwargs = {
    'SafeHopper-v0': {'epochs': 50},
    'init_exploration_steps': 8000,
    'ensemble_hiddens': (20, 20),
}


def argument_parser():
    n_cpus = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num-cores', '-c', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', '-r', type=int, default=3,
                        help='Number of indipendent seeds per experiment.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/ga87zej',
                        help='Define a custom directory for logging.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Define the initial seed.')
    args = parser.parse_args()
    return args


def main(args):

    bench = Benchmark(
        alg_setup,
        env_ids=list(env_specific_kwargs.keys()),
        log_dir=args.log_dir,
        num_cores=args.num_cores,
        num_runs=args.num_runs,
        env_specific_kwargs=env_specific_kwargs,
        use_mpi=True,
        skip_eval=True,
        init_seed=args.seed,
    )
    bench.run()


if __name__ == '__main__':
    loggers.set_level(loggers.ERROR)
    args = argument_parser()
    main(args)

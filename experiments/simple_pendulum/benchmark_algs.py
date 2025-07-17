import argparse
import numpy as np
import psutil

import x_mpsc  # noqa
from x_mpsc.benchmark import Benchmark
import x_mpsc.common.loggers as loggers


alg_setup = {
    'cpo': {'target_kl': [1.0e-4, 5.0e-4, 1.0e-3], 'lam_c': [0.50, 0.90, 0.95]},
    'trpo': {"target_kl": [0.001, 0.01]},
    'lag-trpo': {'target_kl': [1.0e-4, 1.0e-3, 1.0e-2],
                 'lambda_lr': [0.001, 0.01, 0.1]},  # SGD is default
    'sqrl': {'gamma_safe': [0.5, 0.7], 'eps_safe': [0.1, 0.2, 0.3]},
}

env_specific_kwargs = {
    'SimplePendulum-v0': {
        'epochs': 1000,
        'steps_per_epoch': 8000,
        'cost_limit': 0.0,
    },
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

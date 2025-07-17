import os
import json
import copy
import warnings

import numpy as np
from collections import namedtuple, OrderedDict
import matplotlib
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

# local imports
import x_mpsc.common.loggers as loggers
from x_mpsc import plot
import x_mpsc.common.utils as U


def detect_algorithm_setup(
        algorithm_name: str,
        base_dir: str,
) -> dict:
    """ Automatically detect algorithmic setups from configuration files.

    Walks recursively through the provided base directory and opens
    config.json files.

    Returns
    -------
    {'cpo': {'lam_c': [0.9, 0.5, 0.95], 'target_kl': [0.01, 0.005]},
     'trpo': {'target_kl': [0.02, 0.01, 0.005]}
    }
    """
    i = 0
    unified_alg_setup = {}
    experiment_paths = U.get_experiment_paths(base_dir)
    for path in experiment_paths:
        config_file_path = os.path.join(path, 'config.json')
        config = U.get_file_contents(config_file_path)
        # discard irrelevant information from configuration
        config.pop('logger_kwargs', None)
        config.pop('exp_name')
        config.pop('logger', None)
        config.pop('seed')
        config.pop('env_fn')
        config.pop('env_name', None)

        def make_hashable(val):
            if isinstance(val, dict):
                # dictionaries are not hashable, so convert to strings
                v = str(val)
            elif isinstance(val, list) and len(val) == 1:
                # lists are also un-hashable types
                v = val[0]
            else:
                v = val
            return v

        if algorithm_name not in unified_alg_setup:
            # first time that algorithm is found in config files
            value_set_dict = dict()
            for key, val in config.items():
                try:
                    v = make_hashable(val)
                    value_set_dict[key] = set([v, ])
                except:
                    warnings.warn(f"cannot hash: {key}:\n{val}")
            unified_alg_setup[algorithm_name] = value_set_dict

        else:
            # algorihm in detected_alg_setup
            # now add new configs if found
            d = unified_alg_setup[algorithm_name]
            for key, val in config.items():
                try:
                    v = make_hashable(val)
                    d[key].add(v)
                except:
                    warnings.warn(f"cannot hash: {key}:\n{val}")
        i += 1

    # detect all config values that have at least 2 different values
    detected_alg_setup = {}
    print(f'===== Detected Algorithm Setups =====')
    for alg, dic in unified_alg_setup.items():
        print(f'{alg.upper()}:')
        d = detected_alg_setup[alg] = {}
        for param, values in dic.items():
            if len(values) > 1:
                d[param] = list(values)
                print(f'\t- {param}: {d[param]}')
    print(f'=====================================')
    return detected_alg_setup


def get_data(
        exp_analyer: plot.ExperimentAnalyzer,
        query='EpRet/Mean',
        setup: dict = {}
) -> np.ndarray:
    """ fetch data from the experiment analyzers."""
    parameters = tuple(setup.keys())

    try:
        data_dict = exp_analyer.get_data(parameters,  filter=setup)
        # data holds only one key which is the best configuration.
        # extract values as list
        assert len(data_dict) > 0, f'setup: {setup} got no values'
        pd_list = list(data_dict.values())[0]
        data = []
        for pd in pd_list:
            col = pd[query]
            vals = col.values
            data.append(vals)
            a = 1
        # data = np.array([pd[query].values for pd in pd_list])
    except KeyError as e:
        print(data_dict)
        print(f'Key={query} not found in progress.csv(parameters={parameters}, filter={setup})')
        raise e

    return np.array(data)


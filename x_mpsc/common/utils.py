import argparse
import datetime
import os
import sys
import json
from typing import Optional

import gymnasium as gym
import pandas
import numpy as np
import torch as th
from collections import defaultdict
import re
from importlib import import_module


def get_alg_module(alg, *submodules):
    """ inspired by source: OpenAI's baselines."""

    if submodules:
        mods = '.'.join(['x_mpsc', 'algs', alg, *submodules])
        alg_module = import_module(mods)
    else:
        alg_module = import_module('.'.join(['x_mpsc', 'algs', alg, alg]))

    return alg_module


def get_alg_class(alg, env_id, **kwargs):
    """Get the learn function of a particular algorithm."""
    alg_mod = get_alg_module(alg)
    alg_cls_init = getattr(alg_mod, 'get_alg')

    return alg_cls_init(env_id, **kwargs)


def get_experiment_paths(path: str) -> tuple:
    """ Walk through path recursively and find experiment log files.

        Note:
            In a directory must exist a config.json and metrics.json file, such
            that path is detected.

    Parameters
    ----------
    path
        Path that is walked through recursively.

    Raises
    ------
    AssertionError
        If no experiment runs where found.

    Returns
    -------
    list
        Holding path names to directories.
    """
    experiment_paths = []
    for root, dirs, files in os.walk(path):  # walk recursively trough basedir
        config_json_in_dir = False
        metrics_json_in_dir = False
        for file in files:
            if file.endswith("config.json"):
                config_json_in_dir = True
            if file.endswith("progress.csv") or file.endswith("progress.txt"):
                metrics_json_in_dir = True
        if config_json_in_dir and metrics_json_in_dir:
            experiment_paths.append(root)

    assert experiment_paths, f'No experiments found at: {path}'

    return tuple(experiment_paths)

def get_file_contents(file_path: str,
                      skip_header: bool = False):
    """Open the file with given path and return Python object."""
    assert os.path.isfile(file_path), 'No file exists at: {}'.format(file_path)

    if file_path.endswith('.json'):  # return dict
        with open(file_path, 'r') as fp:
            data = json.load(fp)

    elif file_path.endswith('.csv'):
        if skip_header:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        else:
            data = np.loadtxt(file_path, delimiter=",")
        if len(data.shape) == 2:  # use pandas for tables..
            data = pandas.read_csv(file_path)
    elif file_path.endswith('.txt'):
        data = pandas.read_table(file_path)
        a = 1
    else:
        raise NotImplementedError
    return data


def get_learn_function(alg):
    """Get the learn function of a particular algorithm."""
    alg_mod = get_alg_module(alg)
    learn_func = getattr(alg_mod, 'learn')

    return learn_func


def get_env_type(env_id: str):
    """Determines the type of the environment if there is no args.env_type.

    source: OpenAI's Baselines Repository

    Parameters
    ----------
    env_id:
        Name of the gym environment.

    Returns
    -------
    env_type: str
    env_id: str
    """
    all_registered_envs = defaultdict(set)
    reg = gym.envs.registry

    env_spec: gym.envs.registration.EnvSpec
    for env_spec in gym.envs.registry.values():
        try:
            env_type = env_spec.entry_point.split(':')[0].split('.')[-1]
            all_registered_envs[env_type].add(env_spec.id)
        except AttributeError:
            env_type = "default"

    if env_id in all_registered_envs.keys():
        env_type = env_id
        env_id = [g for g in all_registered_envs[env_type]][0]
    else:
        env_type = "default"
        for g, e in all_registered_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id,
            all_registered_envs.keys())

    return env_type, env_id


def get_defaults_kwargs(alg, env_id):
    """ inspired by OpenAI's baselines."""
    env_type, _ = get_env_type(env_id=env_id)

    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        pass
        # warnings.warn(
        #     f'Could not fetch default kwargs for env_type: {env_type}')
        # Fetch standard arguments from locomotion environments
        try:  # fetch from defaults()
            env_type = 'defaults'
            alg_defaults = get_alg_module(alg, 'defaults')
            kwargs = getattr(alg_defaults, env_type)()
        except:
            env_type = 'locomotion'
            alg_defaults = get_alg_module(alg, 'defaults')
            kwargs = getattr(alg_defaults, env_type)()

    return kwargs


def convert_to_string_only_dict(input_dict):
    """
    Convert all values of a dictionary to string objects
    Useful, if you want to save a dictionary as .json file to the disk

    :param input_dict: dict, input to be converted
    :return: dict, converted string dictionary
    """
    converted_dict = dict()
    for key, value in input_dict.items():
        if isinstance(value, dict):  # transform dictionaries recursively
            converted_dict[key] = convert_to_string_only_dict(value)
        elif isinstance(value, type):
            converted_dict[key] = str(value.__name__)
        else:
            converted_dict[key] = str(value)
    return converted_dict


def get_default_args(debug_level=0,
                     env='CartPole-v0',
                     func_name='testing',
                     log_dir='/var/tmp/ga87zej/',
                     threads=os.cpu_count()
                     ):
    """ create the default arguments for program execution
    :param threads: int, number of available threads
    :param env: str, name of RL environment
    :param func_name:
    :param log_dir: str, path to directory where logging files are going to be created
    :param debug_level: 
    :return: 
    """
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    parser = argparse.ArgumentParser(description='This is the default parser.')
    parser.add_argument('--alg', default=os.cpu_count(), type=int,
                        help='Algorithm to use (in case of a RL problem. (default: PPO)')
    parser.add_argument('--threads', default=threads, type=int,
                        help='Number of available Threads on CPU.')
    parser.add_argument('--debug', default=debug_level, type=int,
                        help='Debug level (0=None, 1=Low debug prints 2=all debug prints).')
    parser.add_argument('--env', default=env, type=str,
                        help='Default environment for RL algorithms')
    parser.add_argument('--func', dest='func', default=func_name,
                        help='Specify function name to be testing')
    parser.add_argument('--log', dest='log_dir', default=log_dir,
                        help='Set the seed for random generator')

    args = parser.parse_args()
    args.log_dir = os.path.abspath(os.path.join(args.log_dir,
                                                datetime.datetime.now().strftime(
                                                    "%Y_%m_%d__%H_%M_%S")))
    return args


def get_seed_from_sys_args():
    _seed = 0
    for pos, elem in enumerate(
            sys.argv):  # look if there exists seed=X in _args,
        if len(elem) > 5 and elem[:5] == 'seed=':
            _seed = int(elem[5:])
    return _seed


def normalize(xs,
              axis=None,
              eps=1e-8):
    """ Normalize array along axis
    :param xs: np.array(), array to normalize
    :param axis: int, axis along which is normalized
    :param eps: float, offset to avoid division by zero
    :return: np.array(), normed array
    """
    return (xs - xs.mean(axis=axis)) / (xs.std(axis=axis) + eps)


def mkdir(path):
    """ create directory at a given path
    :param path: str, path
    :return: bool, True if created directories
    """
    created_dir = False
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        created_dir = True
    return created_dir


def to_tensor(x: np.ndarray):
    return th.as_tensor(x, dtype=th.float32)


def to_matrix(x: np.ndarray):
    if x.ndim == 1:
        return x.reshape((-1, 1))
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f"got ndim: {x.ndim}")


def get_experiment_directory(path: str) -> Optional[str]:
    r"""Return the path to the directory holding config.json and progress.csv."""
    latest_directory = sorted(os.listdir(path))[-1]
    directory_path = os.path.join(path, latest_directory)
    # walk recursively trough latest_directory_path
    for root, dirs, files in os.walk(directory_path):
        config_json_in_dir = False
        metrics_json_in_dir = False
        for file in files:
            if file.endswith("config.json"):
                config_json_in_dir = True
            if file.endswith("progress.csv"):
                metrics_json_in_dir = True
        if config_json_in_dir and metrics_json_in_dir:
            return root
    raise ValueError(f"Did not find experiment path.")


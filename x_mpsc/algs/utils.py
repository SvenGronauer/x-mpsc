import argparse
import datetime
import os
import sys
import gymnasium as gym
import numpy as np
from collections import defaultdict
import re
import torch as th
from importlib import import_module

import x_mpsc.common.mpi_tools as mpi

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10, eps=1e-6):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)

    nsteps: (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.
            Also probably don't play with this hyperparameter.
    """
    x = th.zeros_like(b)
    r = b - Avp(x)
    p = r.clone()
    rdotr = th.dot(r, r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    verbose = False

    for i in range(nsteps):
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = Avp(p)
        alpha = rdotr / (th.dot(p, z) + eps)
        x += alpha * p
        r -= alpha * z
        new_rdotr = th.dot(r, r)
        if th.sqrt(new_rdotr) < residual_tol:
            break
        mu = new_rdotr / (rdotr + eps)
        p = r + mu * p
        rdotr = new_rdotr

    return x


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


def get_device(device_str="auto"):
    if device_str == "auto":
        if mpi.num_procs() > 1:
            return th.device("cpu")
        elif th.cuda.is_available():
            th.set_default_tensor_type(th.cuda.FloatTensor)
            return th.device("cuda")
        else:
            return th.device("cpu")
    else:
        try:
            return th.device(device_str)
        except Exception as e:
            print("handling device error:")
            print(e)


def get_learn_function(alg):
    """Get the learn function of a particular algorithm."""
    alg_mod = get_alg_module(alg)
    learn_func = getattr(alg_mod, 'learn')

    return learn_func



def get_flat_gradients_from(model):
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad
            grads.append(g.view(-1))  # flatten tensor and append
    assert grads is not [], 'No gradients were found in model parameters.'

    return th.cat(grads)


def get_flat_params_from(model):
    flat_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            d = param.data
            d = d.view(-1)  # flatten tensor
            flat_params.append(d)
    assert flat_params is not [], 'No gradients were found in model parameters.'

    return th.cat(flat_params)


def hard_update(target, source):
    with th.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def set_flat_grads_to_model(model, grads: th.Tensor):
    # assert isinstance(grads, th.Tensor)
    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = grads[i:i + size]

            # set new gradients
            try:
                param.grad.data = new_values.view(orig_size)
            except AttributeError:
                # AttributeError: 'NoneType' object has no attribute 'data'
                # in case grad is None
                param.grad = new_values.view(orig_size)
            i += size  # increment array position
    assert i == len(grads), f'Lengths do not match: {i} vs. {len(grads)}'


def set_param_values_to_model(model, vals):
    assert isinstance(vals, th.Tensor)
    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i:i + size]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += size  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'


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


def parse_to_sacred_experiment_args(arguments: list, _seed: int,
                                    log_dir: str) -> list:
    """Add seed to sys args if not already defined.

    This brings arguments into the expected structure of the sacred framework.
    Remind: arguments is a list, so this is a call by reference

    Parameters
    ----------
    arguments
        An object holding information of the parsed console arguments. This is a call by reference!
    _seed
        The seed for the random generator.
    log_dir
        Over-write the default BASEDIR by setting a sacred flag.

    Returns
    -------
    list
        The parsed arguments with the expected structure.
    """
    seed_in_args = None
    with_in_args = False
    custom_log_dir = None if log_dir == '' else log_dir

    _args = arguments.copy()  # copy argument list, since this is a call by reference

    # sacred expects the first element of list to be a command
    if len(_args) > 0 and _args[0] == 'with':
        _args = ['run', ] + _args
    if len(_args) == 0:
        _args.append('run')

    for pos, elem in enumerate(_args):  # look if there exists seed=X in _args,
        if len(elem) > 5 and elem[:5] == 'seed=':
            seed_in_args = pos
            value = elem[5:]
            print(
                'WARNING: Seed==={} was found in args, change seed to: {}'.format(
                    value, _seed))
        if len(elem) == 4 and elem == 'with':
            with_in_args = True

    string_argument = 'seed=' + str(_seed)
    if seed_in_args:
        _args[seed_in_args] = string_argument

    else:
        if not with_in_args:
            _args.append('with')
        _args.append(string_argument)

    # over-write the default BASEDIR with custom log dir if provided
    if custom_log_dir:
        _args.append(f'base_dir={custom_log_dir}')

    return _args


def safe_mean(xs):
    """ Calculate mean value of an array safely and avoid division errors
    :param xs: np.array, array to calculate mean
    :return: np.float, mean value of xs
    """
    return np.nan if len(xs) == 0 else float(np.mean(xs))


def safe_std(xs):
    return np.std(xs) if safe_mean(xs) is not np.nan else np.nan

from typing import Optional

import dill
import gymnasium as gym
import time

import argparse
import os
import torch
import numpy as np
import warnings

# local imports
import x_mpsc.common.loggers as loggers
from x_mpsc.algs.terminal_set import TerminalSet
from x_mpsc.algs import core
from x_mpsc.algs.sac.actor_critic import MLPActorCritic
from x_mpsc.common.utils import get_file_contents


def load(
        config: dict,
        env_id: str,
        ckpt: str
):
    env = gym.make(env_id, render_mode="human")
    alg = config.get('alg', 'ppo')
    print("ALG: ", alg)
    if alg == 'mbpo' or alg == 'sac':

        ac = MLPActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            ac_kwargs=config['ac_kwargs']
        )
    else:
        ac = core.ActorCritic(
            actor_type=config['actor'],
            recurrent=config.get('recurrent', False),
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_standardized_obs=config['use_standardized_obs'],
            use_scaled_rewards=config['use_reward_scaling'],
            use_shared_weights=config['use_shared_weights'],
            # weight_initialization=conf['weight_initialization'],
            ac_kwargs=config['ac_kwargs']
        )
    model_path = os.path.join(ckpt, 'torch_save', 'ac.pt')
    ac.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')),
                       strict=False)

    print(f'Successfully loaded model from: {model_path}')
    return ac, env


def play_after_training(
        actor_critic,
        env,
        ckpt: str,
        noise: bool =False,
        ensemble_model = None, #: Optional[EnsembleModel] = None,
        use_mpsc: bool = False,
):
    if not noise:
        actor_critic.eval()  # Set in evaluation mode before playing
    i = 0
    if use_mpsc and ensemble_model is not None:
        # todo sven: load terminal set from disk
        fnp = os.path.join(ckpt, 'state.pkl')
        file = open(fnp, 'rb')
        terminal_set = dill.load(file).get('terminal_set', None)
        assert isinstance(terminal_set,  TerminalSet), f"got wrong class instace."
        print(terminal_set)
        raise NotImplementedError
        mpsc = EnsembleMPSC(
            env=env,
            dynamics_model=ensemble_model,  # fixme sven: dynamics not model here
            horizon=10,
            terminal_set=terminal_set
        )
        mpsc.setup_optimizer()
    else:
        mpsc = None

    # pb.setRealTimeSimulation(1)
    while True:
        done = False
        x, _ = env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, *_ = actor_critic(obs, deterministic=False)
            # if mpsc is not None:
            #     action = mpsc.solve(x, action)
            x, r, terminated, truncated, info = env.step(action)
            costs += info.get('cost', 0.)
            ret += r
            episode_length += 1
            done = terminated or truncated
            time.sleep(1./120)
        i += 1
        print(
            f'Episode {i}\t Return: {ret}\t Length: {episode_length}\t Costs:{costs}')


def random_play(env_id, use_graphics):
    env = gym.make(env_id, render_mode="human")
    i = 0
    rets = []
    costs = []
    TARGET_FPS = 60
    target_dt = 1.0 / TARGET_FPS
    while True:
        i += 1
        done = False
        x, _ = env.reset()
        ts = time.time()
        ret = 0.
        cum_cost = 0.
        ep_length = 0
        while not done:
            ts1 = time.time()
            if hasattr(env, 'safe_controller'):
                action = env.safe_controller(x)
            else:
                action = env.action_space.sample()
            x, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ret += r
            ep_length += 1
            cum_cost += info.get('cost', 0.)
            delta = time.time() - ts1
            if use_graphics:
                if delta < target_dt:
                    time.sleep(target_dt-delta)  # sleep delta time
            # print(f'FPS: {1/(time.time()-ts1):0.1f}')
        rets.append(ret)
        costs.append(cum_cost)
        print(f'Episode {i}\t Return: {ret:0.2f}\t Costs:{cum_cost} Length: {ep_length}'
              f'\t RetMean:{np.mean(rets):0.2f}\t RetStd:{np.std(rets):0.2f} \t CostMean:{np.mean(costs):0.2f}')
        print(f'Took: {time.time()-ts:0.2f}')


if __name__ == '__main__':
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Choose from: {ppo, trpo}')
    parser.add_argument('--env', type=str,
                        help='Example: HopperBulletEnv-v0')
    parser.add_argument('--random', action='store_true',
                        help='Visualize agent with random actions.')
    parser.add_argument('--noise', action='store_true',
                        help='Visualize agent with random actions.')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering.')
    args = parser.parse_args()
    env_id = None
    use_graphics = False if args.no_render else True

    if args.ckpt:
        config_file_path = os.path.join(args.ckpt, 'config.json')
        conf = get_file_contents(config_file_path)
        print('Loaded config file:')
        print(conf)
        env_id = args.env if args.env else conf['env_id']
        try:
            from x_mpsc.models.ensemble import DynamicsModel
            dynamics_model = DynamicsModel.load(args.ckpt)
            print(loggers.colorize("Using X-MPSC!", color="green"))
            ensemble_model = dynamics_model.ensemble_model
        except:
            warnings.warn(f"Did not find an ensemble model...")
            ensemble_model = None

    if args.random:
        # play random policy
        assert env_id or hasattr(args, 'env'), 'Provide --ckpt or --env flag.'
        env_id = args.env if args.env else env_id
        random_play(env_id, use_graphics)
    else:
        assert args.ckpt, 'Define a checkpoint for non-random play!'
        env = gym.make(env_id, render_mode="human")
        alg = conf.get('alg', 'ppo')

        ac, env = load(config=conf, env_id=env_id, ckpt=args.ckpt)

        play_after_training(
            actor_critic=ac,
            env=env,
            ckpt=args.ckpt,
            noise=args.noise,
            ensemble_model=ensemble_model,
            use_mpsc=conf.get('use_mpsc', False)
        )
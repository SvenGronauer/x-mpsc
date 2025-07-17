import argparse

import dill
import ffmpeg
import copy
import glob
import os

import joblib
from tqdm import tqdm

import numpy as np
import torch as th

# local imports
import x_mpsc.common.loggers as loggers
from x_mpsc.common.utils import get_file_contents, get_experiment_directory
from x_mpsc.mpsc import EnsembleMPSC
from x_mpsc.mpsc.wrappers import EnsembleModelCasadiWrapper
from x_mpsc.models.ensemble import DynamicsModel, EnsembleModel
from x_mpsc.envs.twolinkarm import TwoLinkArmEnvPlotWrapper
from x_mpsc.algs.terminal_set import TerminalSet
import x_mpsc.common.mpi_tools as mpi
from x_mpsc.algs.sac.sac import MLPActorCritic
from x_mpsc.play import load


ENV_ID = "TwoLinkArm-v0"


def record_video(
        env,
        ac: MLPActorCritic,
        dynamics_model: DynamicsModel,
        terminal_set: TerminalSet,
        mpsc,
        epoch=0,
        rollout_length=100,
        save_fig = False
):
    loggers.info(f"Play MPBO...")
    pbar = tqdm(range(rollout_length), disable=not mpi.is_root_process())
    eval_env = copy.deepcopy(env)
    x, _ = eval_env.reset()
    mpsc.setup_optimizer()

    nx, nu = env.observation_space.shape[0],  env.action_space.shape[0]
    wrapped_models = [
        EnsembleModelCasadiWrapper(
            dynamics_model, model_idx=m) for m in range(mpsc.M)
    ]

    log_dir = f"/var/tmp/videos/{ENV_ID}"
    video_path = os.path.join(log_dir, "videos")
    print(f"Loaded ckpt from: {ckpt}")
    print(f"Save video to: {video_path}")

    for stage in pbar:
        pbar.set_description(f"{'Plot' if save_fig else 'Recording video'}")

        th_obs = th.as_tensor(x, dtype=th.float32).to(device="cpu")
        u_learn = ac.act(th_obs, deterministic=False)
        u_learn = np.clip(u_learn,  env.action_space.low,  env.action_space.high)
        u = mpsc.solve(x, u_learn)

        # Us = 0.8 * np.ones(( mpsc.nu,  mpsc.horizon-1))
        Us = mpsc.last_Us
        eval_env.plot_current_nominal_trajectory(
            wrapped_casadi_models=wrapped_models,
            ensemble_model=ensemble_model,
            mpsc=mpsc,
            u_learn=u_learn,
            Us=Us,
            log_dir=log_dir,
            epoch=0,
            iteration=stage,
            terminal_set= terminal_set,
            save_fig=True,
            debug=False
        )

        x, r, done, info = eval_env.step(u)
    pbar.close()

    jpg_path = os.path.join(video_path, "*.jpg")

    video_file_path = os.path.join(video_path, f'ep-{ epoch}-movie.mp4')
    files = sorted(glob.glob(jpg_path))
    try:
        ffmpeg.input(jpg_path, pattern_type='glob', framerate=10).output(video_file_path).run(quiet=True)
    except:
        loggers.error("Could not create video file.")

    for file in glob.glob(jpg_path):
        loggers.trace(f"delete: {file}")
        os.remove(file)
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ckpt', '-c', type=str,
        help='Give the path to the model, e.g. /sven/models/best_model.pth')
    parser.add_argument(
        '--horizon', '-n', type=int, default=15,
        help=f'Number of predictive steps.')
    args, unparsed_args = parser.parse_known_args()
    print(f"CKPT: {args.ckpt}")
    loggers.set_level(loggers.INFO)

    if args.ckpt is None:
        base_dir = f"/var/tmp/sven/{ENV_ID}/mbpo/"
        ckpt = get_experiment_directory(base_dir)
    else:
        assert os.path.exists(args.ckpt)
        ckpt = args.ckpt
    config_file_path = os.path.join(ckpt, 'config.json')
    conf = get_file_contents(config_file_path)

    ac, unwrapped_env = load(conf, conf["env_id"], ckpt)
    env = TwoLinkArmEnvPlotWrapper(unwrapped_env)

    print(f"Actor-Critic:\n{ac}")

    fnp = os.path.join(ckpt, 'state.pkl')
    file = open(fnp, 'rb')
    terminal_set = dill.load(file).get('terminal_set', None)
    assert isinstance(terminal_set, TerminalSet), f"got wrong class instace."

    dynamics_model = DynamicsModel.load(ckpt)
    mpsc = EnsembleMPSC(
        env=env,
        dynamics_model=dynamics_model,
        terminal_set=terminal_set,
        horizon=10
    )

    print("*"*55)
    print(dynamics_model.ensemble_model)
    print("*"*55)
    print(terminal_set)
    print("*"*55)

    record_video(env, ac, dynamics_model, mpsc=mpsc, terminal_set=terminal_set)


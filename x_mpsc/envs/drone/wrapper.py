r"""Wrap the DroneEnv environment with useful debug variables.
"""

from typing import Optional, List

import numpy as np
import gymnasium as gym
import copy
import os
import matplotlib.pyplot as plt
import torch

# local imports
from x_mpsc.common import loggers
from x_mpsc.envs.drone.drone import DroneEnv
from x_mpsc.common.sets import EllipsoidalSet, BoxSet
from x_mpsc.algs.terminal_set import TerminalSet
# from x_mpsc.mpsc import EnsembleMPSC
from x_mpsc.mpsc.wrappers import EnsembleModelCasadiWrapper
from x_mpsc.algs.mbpo.env_wrapper import PredictEnv
# from x_mpsc.models.ensemble import EnsembleModel

colors = ['blue', 'orange', 'cyan', 'green', 'purple']


class DronePlotWrapper(gym.Wrapper):

    def __init__(self, env: DroneEnv):
        super().__init__(env)

    @torch.no_grad()
    def plot_current_nominal_trajectory(
            self,
            wrapped_casadi_models: List[EnsembleModelCasadiWrapper],
            mpsc, # EnsembleMPSC
            u_learn: np.ndarray,
            Us: np.ndarray,
            log_dir: str = "/var/tmp/fakepath",
            epoch: int = 0,
            iteration: int = 0,
            save_fig: bool = True,
            terminal_set: Optional[TerminalSet] = None,
            debug: bool = False
    ):
        PLOT_DIMS = (3, 4)
        state_space_box = BoxSet(from_space=self.env.observation_space)
        eval_env = copy.deepcopy(self.env)
        x = copy.deepcopy(eval_env.state)
        N = Us.shape[1] + 1
        M = len(wrapped_casadi_models)
        PLOT_MODELS = int(min(5, M))
        nx = self.observation_space.shape[0]
        nu = self.action_space.shape[0]
        Xs = np.zeros((M, nx, N))
        for i in range(M):
            Xs[i, :, 0] = x.copy()

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
        color_maps = ['Oranges', 'Greens', 'Purples', 'Blues', 'Reds', ]
        ts = np.arange(N)
        ax1.set_xlabel('Roll [rad]')
        ax1.set_ylabel('Pitch [rad]')
        ax1.set_title("State Space")
        c = 'black' if mpsc is not None and mpsc.feasible else "red"
        state_space_box.draw(ax1, color=c, dims=PLOT_DIMS)

        # == Ground truth
        X_true = np.zeros((nx, N))
        X_true[:, 0] = x
        initial_env_state = x.copy()
        for k in range(0, N - 1):
            if (k == 0):
                # get environment RGB rendering
                eval_env.last_u = Us[:, k]

                # todo sven: use bullet image
                #env_image = eval_env.render(mode="rgb_array")
                # ax3.imshow(env_image)
            x, *_ = eval_env.step(Us[:, k].flatten())
            X_true[:, k + 1] = x

        # if mpi.is_root_process():
        ax1.set_xlim(eval_env.observation_space.low[PLOT_DIMS[0]]-0.05,
                     eval_env.observation_space.high[PLOT_DIMS[0]]+0.05)
        ax1.set_ylim(eval_env.observation_space.low[PLOT_DIMS[1]]-0.05,
                     eval_env.observation_space.high[PLOT_DIMS[1]]+0.05)

        if terminal_set is not None:
            terminal_set.draw_convex_hull(
                ax1, color='grey', linestyle='solid', dims=PLOT_DIMS)

        # forward rollout
        for k in range(N-1):  # horizon
            for m in range(M):
                obs = Xs[m, :, k]
                acs = Us[:, k]

                model = wrapped_casadi_models[m]
                assert hasattr(model, 'f_discrete')
                x_next = model.f_discrete(obs, acs).full().flatten()

                # x_next, *_ = virtual_env.step(obs, acs, deterministic=True, model_idx=m)
                Xs[m, :, k+1] = x_next

        # print(f"Terminal Set with Q_term=\n{terminal_set.ellipse.Q}")
        for m in range(PLOT_MODELS):
            error_ellipse = EllipsoidalSet(1e-12 * np.eye(nx), np.zeros(nx))
            cmap = plt.get_cmap(color_maps[m])
            wrapped_model = wrapped_casadi_models[m]

            for k in range(N-1):  # horizon
                #print(f"------- k={k} -------")
                A_k = wrapped_model.get_df_dx(Xs[m, :, k], Us[:, k]).full()
                B_k = wrapped_model.get_df_du(Xs[m, :, k], Us[:, k]).full()
                if mpsc is not None:
                    F_k = A_k + B_k @ mpsc.K_numpy
                else:
                    F_k = A_k
                Q_w = wrapped_model.get_Q(Xs[m][:, k], Us[:, k]).full()
                error_ellipse.transform(A=F_k)
                error_ellipse.add_set(other=EllipsoidalSet(Q_w, np.zeros(nx)))
                error_ellipse.set_center(Xs[m][:, k+1].flatten())
                error_ellipse.draw(ax1, color=cmap(0.7*k / N+0.3), dims=PLOT_DIMS)

            ax1.scatter(Xs[m, PLOT_DIMS[0]], Xs[m, PLOT_DIMS[1]], c=ts, cmap=cmap, marker="x")

        """action plots"""
        # color1 = colors[0] if mpsc.feasible else "red"
        # color2 = colors[1] if mpsc.feasible else "red"
        # ax2.step(np.arange(Us.size), u_learn * np.ones_like(Us[0]),  where='post', color=color1)
        # ax2.step(np.arange(Us.size), Us[0, mpsc.k_inf] * np.ones_like(Us[0]), where='post', color=colors[3])
        # ax2.step(np.arange(Us.size), Us[0], where='post', color=colors[1])
        #
        # # print constraints
        # for j in range(1):
        #     if mpsc.last_U_bounds is not None:
        #         U_bounds = mpsc.last_U_bounds[j]
        #         sequence_length = U_bounds.shape[1]
        #         assert sequence_length == Us.shape[1], 'action sequence mismatch'
        #         ax2.step(np.arange(sequence_length), U_bounds[0], where='post', color='black')
        #         ax2.step(np.arange(sequence_length), -U_bounds[1], where='post', color='black')

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Force [N]')
        ax2.set_title("Action")
        ax2.set_ylim(-1.1, 1.1)

        # ground truths
        ax1.plot(X_true[PLOT_DIMS[0]], X_true[PLOT_DIMS[1]], marker="x",
                 linestyle='-', color='blue')

        if save_fig:
            video_path = os.path.join(log_dir, "videos")
            os.makedirs(video_path, exist_ok=True)
            # print(f"saved to: {video_path}") if debug else None
            plt.savefig(os.path.join(video_path, f"ep{epoch+1}-{str(iteration).zfill(3)}.jpg"))
            loggers.debug(f"Saved figure: ep{epoch+1}-{iteration}.jpg")
        else:
            plt.show()
        plt.close(fig)

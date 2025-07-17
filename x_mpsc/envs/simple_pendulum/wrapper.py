r"""The Simple Pendulum from Underactuated Robotics (Russ Tedrake).

http://underactuated.mit.edu/pend.html
"""
from __future__ import annotations
from typing import Optional, List

import numpy as np
import gymnasium as gym
import copy
import os
import matplotlib.pyplot as plt
import torch

# local imports
from x_mpsc.common import loggers
from x_mpsc.envs.simple_pendulum.pendulum import SimplePendulumEnv
from x_mpsc.common.sets import EllipsoidalSet, BoxSet
from x_mpsc.algs.terminal_set import TerminalSet
from x_mpsc.mpsc.wrappers import EnsembleModelCasadiWrapper
from x_mpsc.algs.mbpo.env_wrapper import PredictEnv


class SimplePendulumPlotWrapper(gym.Wrapper):

    def __init__(self, env: SimplePendulumEnv):
        super().__init__(env)

    @torch.no_grad()
    def plot_current_nominal_trajectory(
            self,
            wrapped_casadi_models: List[EnsembleModelCasadiWrapper],
            mpsc,  # EnsembleMPSC
            u_learn: np.ndarray,
            Us: np.ndarray,
            log_dir: str,
            epoch: int,
            iteration: int,
            save_fig: bool = True,
            terminal_set: Optional[TerminalSet] = None,
            debug: bool = False
    ):
        eval_env = copy.deepcopy(self.env)
        state_space_box = BoxSet(from_space=self.env.observation_space)
        x = copy.deepcopy(eval_env.state)
        N = Us.shape[1] + 1
        M = len(wrapped_casadi_models)
        PLOT_MODELS = int(min(5, M))
        nx = 2
        nu = 1
        Xs = np.zeros((M, nx, N))
        for i in range(M):
            Xs[i, :, 0] = eval_env.state

        # if mpi.is_root_process():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 7))
        color_maps = ['Oranges', 'Greens', 'Purples', 'Blues', 'Reds', ]
        ts = np.arange(N)
        ax1.set_xlabel('Position [m]')
        ax1.set_ylabel('Velocity [m/s]')
        ax1.set_title("Tube-based Planning")

        c = 'black'
        if mpsc is not None:
            c = 'red' if mpsc.is_failure else ("black" if mpsc.feasible else "orange")
        state_space_box.draw(ax1, color=c)

        # == Ground truth
        X_true = np.zeros((nx, N))
        X_true[:, 0] = x
        initial_env_state = x.copy()
        for k in range(N - 1):
            if (k == 0):
                # get environment RGB rendering
                eval_env.last_u = Us[:, k]
                env_image = eval_env.render(mode="rgb_array")
                ax3.imshow(env_image)
            x, *_ = eval_env.step(Us[:, k].flatten())
            X_true[:, k + 1] = x

        # if mpi.is_root_process():
        ax1.set_xlim(eval_env.observation_space.low[0]-0.5, eval_env.observation_space.high[0]+0.5)
        ax1.set_ylim(eval_env.observation_space.low[1]-0.5, eval_env.observation_space.high[1]+0.5)

        if mpsc is not None and terminal_set is not None:
            terminal_set.draw_convex_hull(ax1, color='blue', linestyle='solid')

        # forward rollout
        for k in range(N-1):  # horizon
            for m in range(M):
                obs = Xs[m, :, k]
                acs = Us[:, k]

                model = wrapped_casadi_models[m]
                assert hasattr(model, 'f_discrete')
                x_next = model.f_discrete(obs, acs).full().flatten()
                Xs[m, :, k+1] = x_next

        for m in range(PLOT_MODELS):
            error_ellipse = EllipsoidalSet(1e-12 * np.eye(nx), np.zeros(nx))
            cmap = plt.get_cmap(color_maps[m])
            wrapped_model = wrapped_casadi_models[m]

            for k in range(0, N-1):  # horizon
                A_k = wrapped_model.get_df_dx(Xs[m, :, k], Us[:, k]).full()
                B_k = wrapped_model.get_df_du(Xs[m, :, k], Us[:, k]).full()
                if mpsc is not None:
                    F_k = A_k + B_k @ mpsc.K_numpy
                else:
                    F_k = A_k
                Q_w = wrapped_model.get_Q(Xs[m][:, k], Us[:, k]).full()
                error_ellipse.transform(A=F_k)
                error_ellipse.add_set(other=EllipsoidalSet(Q_w, np.zeros(2)))
                error_ellipse.set_center(Xs[m][:, k+1].flatten())
                error_ellipse.draw(ax1, color=cmap(0.7*k / N+0.3))

            ax1.scatter(Xs[m, 0], Xs[m, 1], c=ts, cmap=cmap, marker="x")

            # if mpsc.last_Xs is not None and mpsc.feasible:
            #     idx = m * self.horizon
            #     Xs = mpsc.last_Xs[:, ]
            #     ax1.scatter(Xs[m, 0], Xs[m, 1], c=ts, cmap=cmap, marker="x")
            # Xs = mpsc.last_Xs


        """action plots"""
        def colorize(color: str):
            if mpsc is None:
                return color
            return 'red' if mpsc.is_failure else (color if mpsc.feasible else "orange")

        u_learn_clipped = np.clip(u_learn, -eval_env.max_torque, eval_env.max_torque)
        if mpsc is not None and mpsc.k_inf > 1:
            ax2.step(np.arange(Us.size), Us[0, mpsc.k_inf] * np.ones_like(Us[0]),
                     where='post', color=colorize("orange"))
        ax2.step(np.arange(Us.size), u_learn_clipped * np.ones_like(Us[0]),  where='post', color="blue")
        ax2.step(np.arange(Us.size), Us[0], where='post', color=colorize("green"))

        # print constraints
        for j in range(1):  # todo; self.M)
            if mpsc is not None and mpsc.last_U_bounds is not None:
                U_bounds = mpsc.last_U_bounds[j]
                sequence_length = U_bounds.shape[1]
                assert sequence_length == Us.shape[1], 'action sequence mismatch'
                ax2.step(np.arange(sequence_length), U_bounds[0], where='post', color='black')
                ax2.step(np.arange(sequence_length), -U_bounds[1], where='post', color='black')

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Force [N]')
        ax2.set_title("Action")
        ax2.set_ylim(1.33 * eval_env.action_space.low, 1.33 * eval_env.action_space.high)

        ax1.plot(X_true[0], X_true[1], marker="x", linestyle='-',
                 color='blue')

        if save_fig:
            video_path = os.path.join(log_dir, "videos")
            os.makedirs(video_path, exist_ok=True)
            # print(f"saved to: {video_path}") if debug else None
            plt.savefig(os.path.join(video_path, f"ep{epoch+1}-{str(iteration).zfill(3)}.jpg"))
            loggers.debug(f"Saved figure: ep{epoch+1}-{iteration}.jpg")
        else:
            plt.show()
        plt.close(fig)

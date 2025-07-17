from typing import Optional, List

import numpy as np
import gymnasium as gym
import copy
import os
import matplotlib.pyplot as plt
import torch

# local imports
# from x_mpsc.models.ensemble import EnsembleModel
from x_mpsc.mpsc.wrappers import EnsembleModelCasadiWrapper
from x_mpsc.envs.twolinkarm.twolinkarm import TwoLinkArmEnv
from x_mpsc.common.sets import EllipsoidalSet, BoxSet
# from x_mpsc.mpsc import EnsembleMPSC
from x_mpsc.algs.terminal_set import TerminalSet


class TwoLinkArmEnvPlotWrapper(gym.Wrapper):

    def __init__(self, env: TwoLinkArmEnv):
        super().__init__(env)

    @torch.no_grad()
    def plot_current_nominal_trajectory(
            self,
            wrapped_casadi_models: List[EnsembleModelCasadiWrapper],
            mpsc, #: EnsembleMPSC,
            u_learn: np.ndarray,
            Us: np.ndarray,
            log_dir: str,
            epoch: int,
            iteration: int,
            save_fig: bool = True,
            terminal_set: Optional[TerminalSet] = None,
            debug: bool = False
    ):
        eval_env = copy.deepcopy(self)
        PLOT_DIMS = (0, 1)
        PLOT_MODELS = 5
        x = eval_env.get_observation()
        initial_env_state = x.copy()
        state_space_box = BoxSet(from_space=self.env.observation_space)

        xy_target = x[0:2] - x[2:4]

        N = mpsc.horizon
        M = len(wrapped_casadi_models)
        nx = eval_env.observation_space.shape[0]
        state_space_box = BoxSet(from_space=self.env.observation_space)

        # if mpi.is_root_process():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
        color_maps = ['Oranges', 'Greens', 'Purples', 'Blues', 'Reds', ]
        ts = np.arange(N)
        ax1.set_xlabel('x-Position [m]')
        ax1.set_ylabel('y-Position [m]')
        ax1.set_title("State Space")
        c = 'black' if mpsc.feasible else "red"
        state_space_box.draw(ax1, color=c, dims=PLOT_DIMS)

        # XY end-effector target
        ax1.scatter(xy_target[0], xy_target[1], marker="*", color='yellow', edgecolors="black")

        Xs = np.zeros((M, nx, N))  # plot only x-y positions
        for i in range(M):
            Xs[i, :, 0] = x
        # == Ground truth
        X_true = np.zeros((nx, N))
        X_true[:, 0] = x
        for k in range(0, N - 1):
            if (k == 0):
                # get environment RGB rendering
                env_image = eval_env.render(mode="rgb_array")
                ax3.imshow(env_image)
            x, *_ = eval_env.step(Us[:, k].flatten())
            X_true[:, k + 1] = x

        ax1.plot(X_true[0], X_true[1], marker="x", linestyle='-',
                 color='blue')
        ax1.scatter(initial_env_state[0], initial_env_state[1], marker="*",
                    color='yellow', edgecolors="black")
        ax1.set_xlim(eval_env.observation_space.low[0]-0.1, eval_env.observation_space.high[0]+0.1)
        ax1.set_ylim(eval_env.observation_space.low[1]-0.1, eval_env.observation_space.high[1]+0.1)

        if terminal_set is not None:
            terminal_set.draw_convex_hull(ax1, color='black', linestyle='dashed')

        # forward rollouts
        for k in range(N-1):  # horizon
            for m in range(M):
                obs = Xs[m, :, k]
                acs = Us[:, k]

                model = wrapped_casadi_models[m]
                assert hasattr(model, 'f_discrete')
                x_next = model.f_discrete(obs, acs).full().flatten()

                # x_next, *_ = virtual_env.step(obs, acs, deterministic=True, model_idx=m)
                Xs[m, :, k+1] = x_next

        for m in range(PLOT_MODELS):
            error_ellipse = EllipsoidalSet(1e-12 * np.eye(nx), np.zeros(nx))
            # state_space_box.draw(ax1, color='black')
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
            ax1.scatter(Xs[m, PLOT_DIMS[0]], Xs[m, PLOT_DIMS[1]], c=ts,
                        cmap=cmap, marker="x")

        """action plots"""
        u_learn_clipped = np.clip(u_learn, -eval_env.force_factor, eval_env.force_factor)
        color1 = "royalblue" if mpsc.feasible else "red"
        color2 = "forestgreen" if mpsc.feasible else "red"
        ax2.step(np.arange(N-1), u_learn_clipped[0] * np.ones(N-1),  where='post', color=color1)
        ax2.step(np.arange(N-1), u_learn_clipped[1] * np.ones(N-1),  where='post', color=color2)
        print(f"Us: {Us.shape}")
        ax2.step(np.arange(N-1), Us[0], where='post', color="orange")
        ax2.step(np.arange(N-1), Us[1], where='post', color="orange")

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Torque [Nm]')
        ax2.set_title("Action")
        ax2.set_ylim(-1.1 * eval_env.force_factor, 1.1 * eval_env.force_factor)

        video_path = os.path.join(log_dir, "videos")
        if save_fig:
            os.makedirs(video_path, exist_ok=True)
            plt.savefig(os.path.join(video_path,
                                     f"ep{epoch}-{str(iteration).zfill(3)}.jpg"))
        else:
            plt.show()
        # loggers.debug(f"Saved figure: ep{epoch}-{iteration}.jpg")
        plt.close(fig)

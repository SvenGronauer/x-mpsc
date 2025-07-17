r"""Neural network-based multi-model learning-based Model.predictive control
for nonlinear systems, i.e.. Simple Pendulum..

Author: Sven Gronauer (sven.gronauer@tum.de)
Created: 07.09.2022
"""
from __future__ import annotations

import time
from typing import Optional
import numpy as np
import gymnasium as gym
import casadi as cs
import copy

# local imports
import x_mpsc.common.loggers as loggers
from x_mpsc.models import DynamicsModel
from x_mpsc.mpsc.wrappers import EnsembleModelCasadiWrapper
from x_mpsc.common.sets import BoxSet
from x_mpsc.mpsc.utils import casadi_add_two_ellipsoids, \
    casadi_ellipsoid_in_polytope_constraint, bring_to_matrix, \
    get_reduced_box_from_ellipsoid
from x_mpsc.algs.terminal_set import TerminalSet


class EnsembleMPSC:
    r"""Neural-network-based Multi-Model (MuMo) MPC for linear systems."""
    def __init__(self,
                 env: gym.Env,
                 dynamics_model: DynamicsModel,
                 horizon: int = 10,
                 debug: bool = False,
                 terminal_set: Optional[TerminalSet] = None,
                 feedback_factor: float = 0.5,
                 ):
        self.env = env
        self.dynamics_model = dynamics_model
        self.M = dynamics_model.ensemble_size
        self.horizon = horizon
        self.terminal_set = terminal_set
        self.debug = debug
        self.time_step = 0
        self.last_Xs = None
        self.last_Us = None
        self.last_U_bounds = None
        self.k_inf = 0
        self.feasible = False
        self.is_failure = False

        self.nx, self.nu = env.observation_space.shape[0], env.action_space.shape[0]
        self.K_numpy = -feedback_factor * np.ones((self.nu, self.nx))
        self.K = cs.MX(self.K_numpy)
        self.warm_start = True

        self.state_space_box = BoxSet(from_space=self.env.observation_space)
        self.action_space_box = BoxSet(from_space=self.env.action_space)
        self.is_setup = False

    def setup_optimizer(self):
        r"""Instantiate the MPC problem."""
        loggers.debug('Setup Optimizer...')
        assert self.terminal_set is not None, "Empty terminal set in MPSC"
        self.opti = cs.Opti()

        # ---- decision variables ---------
        N = self.horizon
        # self.Xs = Xs = self.opti.variable(self.M, nx, N)  # state trajectory
        self.Us = Us = self.opti.variable(self.nu, N-1)  # control trajectory (throttle)

        # self.Xs = [self.opti.variable(self.nx, N) for _ in range(self.M)]
        self.Xs = self.opti.variable(self.nx, N*self.M)

        self.U_bounds = [self.opti.variable(2*self.nu, N - 1) for _ in range(self.M)]
        # cs.MX(self.action_space_box.b)

        # ---- initial state and reference ---------
        self.x_init = self.opti.parameter(self.nx, 1)
        # Certified input.
        self.u_tilde = self.opti.variable(self.nu, 1)
        # Desired input.
        self.u_L = self.opti.parameter(self.nu, 1)
        # terminal set (as ellipsoid)
        # self.terminal_Q = self.opti.parameter(self.nx, self.nx)
        # self.terminal_c = self.opti.parameter(self.nx, 1)
        # auxilary terminal variables
        # self.ys = [self.opti.variable(self.nx, 1) for _ in range(self.M)]

        dynamics = []
        casadi_wrapped_models = []
        for model_idx in range(self.M):
            wrapped_model = EnsembleModelCasadiWrapper(
                self.dynamics_model, 
                model_idx=model_idx
            )
            casadi_wrapped_models.append(wrapped_model)
            assert hasattr(wrapped_model, 'f_discrete')
            F = wrapped_model.f_discrete  # <-- this is the NN forward pass
            dynamics.append(F)

        # ---- forward dynamics and cost -----------
        # cost = 0.0
        for m in range(self.M):
            Xs = self.Xs #[m]
            idx = m * N
            f = dynamics[m]
            for k in range(N-1):
                x_next = f(Xs[:, idx+k], Us[:, k])
                self.opti.subject_to(Xs[:, idx+k+1] == x_next)

        # ---- objective          ---------
        # self.opti.minimize(cost)  # sum of squares

        cost = (self.u_L - self.u_tilde).T @ (self.u_L - self.u_tilde)

        # ---- state constraints -----------
        H_x = cs.MX(self.state_space_box.A)
        d_x = cs.MX(self.state_space_box.b)
        H_u = cs.MX(self.action_space_box.A)
        d_u = cs.MX(self.action_space_box.b)
        K = self.K
        zeros = cs.MX.zeros(H_x.shape[1], 1)

        for j, casadi_model in enumerate(casadi_wrapped_models):
            Q = 1e-12 * cs.MX.eye(self.nx)
            idx = j * N
            Xs = self.Xs # [j]
            self.opti.subject_to(Xs[:, idx+0] == self.x_init)

            for k in range(N-1):
                U_bound = self.U_bounds[j]
                b_reduced, Hc = get_reduced_box_from_ellipsoid(c=Us[:, k], Q=K@Q@K.T, H=H_u, d=d_u)
                U_bound[:, k] = b_reduced
                self.opti.subject_to(Hc <= U_bound[:, k])

                A_k = casadi_model.get_df_dx(Xs[:, idx+k], Us[:, k])
                B_k = casadi_model.get_df_du(Xs[:, idx+k], Us[:, k])
                F_k = A_k + B_k @ self.K
                Q_w = casadi_model.get_Q(Xs[:, idx+k], Us[:, k])
                Q = F_k @ Q @ F_k.T
                _, Q = casadi_add_two_ellipsoids(zeros, zeros, Q, Q_w)

                self.opti.subject_to(
                    casadi_ellipsoid_in_polytope_constraint(Xs[:, idx+k+1], Q, H_x, d_x) <= 0)

            """ --- new terminal constraints"""
            H_term = cs.MX(self.terminal_set.A)
            d_term = cs.MX(self.terminal_set.b)
            f_idx = (j+1) * N - 1
            # term_constraint = casadi_ellipsoid_in_polytope_constraint(Xs[:, f_idx], Q, H_term, d_term) <= 0
            # self.opti.subject_to(term_constraint)

            """ terminal soft constraint """
            dist_to_term_set = casadi_ellipsoid_in_polytope_constraint(Xs[:, f_idx], Q, H_term, d_term)
            d = cs.fmax(dist_to_term_set, 0)
            cost += 1e6 * d.T @ d

        # ---- input/action constraints -----------
        u_low = self.env.action_space.low
        u_high = self.env.action_space.high

        self.opti.subject_to(self.u_tilde == Us[:, 0])
        self.opti.subject_to(self.opti.bounded(u_low, Us, u_high))

        self.opti.minimize(cost)

        # Create solver (IPOPT solver as of this version).
        opts = {
            "ipopt.print_level": 0,  # change to 5 for detailed console prints
            "ipopt.sb": "yes",
            "ipopt.max_iter": 250,
            "print_time": 0  # prints the time taken for IP-Opt
        }

        # todo sven: track number of iterations used for IPopt
        self.opti.solver('ipopt', opts)
        # print(self.opti)
        self.is_setup = True
        loggers.info('Done setup optimizer.')

    def solve(
            self,
            obs: np.ndarray,
            uncertified_input: np.ndarray,  # u_L
    ) -> np.ndarray:
        """Solve the MPC optimization problem for a given observation and uncertified input.
        """
        assert self.is_setup, f"Please call setup_optimizer() first!"
        dt = time.time()
        loggers.debug(f'Solve_optimization with X:\n{obs} ')
        self.feasible = False
        self.opti.set_value(self.x_init, obs)
        self.opti.set_value(self.u_L, uncertified_input)

        # Initial guess for optimization problem.
        has_prior_values = self.last_Xs is not None
        if self.warm_start and has_prior_values:
            # Shift previous solutions by 1 step.
            x_guess = copy.deepcopy(self.last_Xs)
            u_guess = copy.deepcopy(self.last_Us)
            u_guess[:, :-1] = u_guess[:, 1:]
            x_guess[:, :-1] = x_guess[:, 1:]
            self.opti.set_initial(self.Xs, x_guess)
            self.opti.set_initial(self.Us, u_guess)

        # Solve the optimization problem.
        try:
            sol = self.opti.solve()
            self.last_Xs = sol.value(self.Xs)
            self.last_Us = bring_to_matrix(sol.value(self.Us))
            self.last_U_bounds = [sol.value(self.U_bounds[j]) for j in range(self.M)]
            action = self.last_Us[:, 0].flatten()

            cost_val = sol.value(self.opti.f)
            action_cost = np.linalg.norm(uncertified_input-action)**2
            diff = np.linalg.norm(action_cost - cost_val)
            self.feasible = True if diff < 1e-6 else False
        except RuntimeError:
            action = None
            loggers.error("Did not find a solution")
            c = 'green' if self.feasible else 'red'
            msg_feasible = f'Feasible? {loggers.colorize(str(self.feasible), c)}'
            loggers.info(f'certify action: u~: {action} '
                         f'u_L: {uncertified_input} {msg_feasible}')

        self.is_failure = False
        if self.feasible:
            self.k_inf = 0
        else:
            self.k_inf += 1
            if (self.k_inf < self.horizon-1) and has_prior_values:
                v = self.last_Us[:, self.k_inf].flatten()
                errors = []
                for i in range(self.M):
                    idx = self.k_inf + i * self.horizon
                    errors.append(obs - self.last_Xs[:, idx])
                error = np.mean(np.array(errors), axis=0)
                action = v + self.K_numpy @ error
                loggers.info(f"k_inf={self.k_inf}: Reuse old solution K(x-z)={self.K_numpy @ error}")
                loggers.trace(f'-- action: {action}')
            else:
                #  action = self.lqr_gain @ obs
                self.is_failure = True
                action = self.last_Us[:, 0].flatten() if self.last_Us is not None else uncertified_input
                loggers.warn("MPSC ran out of backup actions")
            self.k_inf = int(min(self.k_inf, self.horizon-2))
        delta = time.time() - dt
        loggers.debug(f'MPSC action: in={uncertified_input} out={action} '
                      f'diff={uncertified_input-action}\ttook: {delta:0.2f}s')
        return action


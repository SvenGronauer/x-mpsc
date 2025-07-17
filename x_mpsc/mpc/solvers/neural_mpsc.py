r"""Solver classes and functions for safe reinforcement learning.

"""
import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
import casadi as cs
from copy import deepcopy
from scipy import linalg
from munch import Munch

import torch as th
import torch.nn as nn

# local imports:
import x_mpsc.common.loggers as loggers
import x_mpsc.mpc.core as mpc_core
from x_mpsc.models.base import DynamicsModel
import x_mpsc.models
from x_mpsc.mpc.constraints import create_ConstraintList_from_list, \
    BoxConstraint, QuadraticConstraint, ConstrainedVariableType, \
    LinearConstraint


class NeuralMPSC(mpc_core.Solver):
    r"""Neural Model Predictive Safety Certification Solver (MPSC)."""
    def __init__(self,
                 env: gym.Env,
                 model: DynamicsModel,
                 # model: x_mpsc.models.SymbolicModel,  #: SymbolicModel,
                 omega: float,
                 horizon: int = 10,
                 debug: bool = False
                 ):
        # Call super class
        super(NeuralMPSC, self).__init__(
            env, model, debug)

        self.act_dim = env.action_space.shape[0]
        self.nu = self.act_dim
        self.obs_dim = env.observation_space.shape[0]
        self.nx = self.obs_dim

        np.set_printoptions(precision=3)

        self.omega = omega
        self.z_prev = None
        self.v_prev = None
        self.u_tilde_prev = None
        self.prev_action = None

        self.horizon = horizon

        self.results_dict = {}

        self.lqr_gain = np.ones(2)  # todo

        self.k_inf = 0
        self.time_step = 0

        self._create_dynamics_function_from_torch_model()
        
        # Setup constraints based on observation and action space of env
        self._setup_constraints()

        # Now that constraints are defined, setup the optimizer.
        self.setup_optimizer()

        self.results_dict = {}
        self.setup_results_dict()

    def _create_dynamics_function_from_torch_model(self) -> None:
        network = self.model
        assert isinstance(network, nn.Module), \
            f"Expecting model to be torch module."
        with th.no_grad():
            out_test = self.model(th.ones(2), th.ones(1)).numpy()

        w0 = network.net[0].weight.detach().numpy()
        b0 = network.net[0].bias.detach().numpy()
        w1 = network.net[2].weight.detach().numpy()
        b1 = network.net[2].bias.detach().numpy()
        print(*w0.shape)
        print(w0.dtype)

        cas_x = cs.MX.sym('x', self.obs_dim, 1)
        cas_u = cs.MX.sym('u', self.act_dim, 1)
        cas_w0 = cs.MX.sym('w0', *w0.shape)
        print(f'cas_w0: {cas_w0.shape}')
        cas_b0 = cs.MX.sym('b0', *b0.shape)
        cas_w1 = cs.MX.sym('w1', *w1.shape)
        cas_b1 = cs.MX.sym('b1', *b1.shape)
        self.dynamics_nn_func = cs.Function(
            'net', [cas_x, cas_u, cas_w0, cas_b0, cas_w1, cas_b1],
            [cs.mtimes(cas_w1, cs.tanh(cs.mtimes(cas_w0, cs.vertcat(cas_x, cas_u)) + cas_b0)) + cas_b1]
        )
        self.nn_parameters = (w0, b0, w1, b1)
        casadi_func_output = self.dynamics_nn_func(np.ones((2,1)), np.ones((1,1)), *self.nn_parameters)
        print('*' * 55)
        print(f'Casadi: {casadi_func_output} vs torch: {out_test}')
        np.allclose(casadi_func_output, out_test)
        loggers.info('Network successfully transformed into casadi function.')

    def certify_action(self,
                       obs: np.ndarray,
                       uncertified_input: np.ndarray,  # u_L
                       target_state: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, bool]:
        """Check if system stays safe if proposed action is applied. Returns
        a safe action otherwise.

        Algorithm 1 from Wabsersich 2019.
        """
        u_L = uncertified_input
        self.results_dict['obs'].append(obs)
        self.results_dict['learning_actions'].append(u_L)

        action, feasible = self.solve_optimization(
            obs, u_L, target_state)
        c = 'green' if feasible else 'red'
        msg_feasible = f'Feasible? {loggers.colorize(str(feasible), c)}'
        loggers.info(f'certify action: u~: {action} u_L: {u_L} {msg_feasible}')

        self.results_dict['feasible'].append(feasible)
        # print(self.results_dict)

        if feasible:
            self.k_inf = 0
        else:
            self.k_inf += 1
            if (self.k_inf <= self.horizon - 1 and
                    self.z_prev is not None and
                    self.v_prev is not None):
                # clip kinf (in case we use a horizon update scheme
                if self.k_inf >= self.v_prev.shape[1]:
                    self.k_inf = self.v_prev.shape[1] - 1

                err = obs - target_state - self.z_prev[:, self.k_inf]
                
                loggers.trace(f'-- z_prev.shape: {self.z_prev.shape}')
                loggers.trace(f'-- Error: {err}')
                loggers.trace(f'-- Adjust: {self.lqr_gain @ err}')
                loggers.trace(f'-- v_prev.shape: {self.v_prev.shape}')
                loggers.trace(f'-- v_prev: {self.v_prev[:, self.k_inf]}')
                action = self.v_prev[:, self.k_inf] + self.lqr_gain @ err
                loggers.trace(f'-- action: {action}')
            else:
                action = self.lqr_gain @ obs

        action = np.asarray(action).reshape(u_L.shape)
        self.results_dict['kinf'].append(self.k_inf)

        action_diff = np.linalg.norm(u_L - action)
        loggers.debug(f'final action: {action}')
        self.results_dict['actions'].append(action)
        self.results_dict['corrections'].append(action_diff)

        return action, feasible

    def close_results_dict(self):
        """Cleanup the rtesults dict and munchify it.

        """
        self.results_dict['obs'] = np.vstack(self.results_dict['obs'])
        # self.results_dict['uncertified_obs'] = np.vstack(self.results_dict['uncertified_obs'])
        # self.results_dict['uncertified_actions'] = np.vstack(self.results_dict['uncertified_actions'])
        self.results_dict['actions'] = np.vstack(self.results_dict['actions'])
        self.results_dict['learning_actions'] = np.vstack(
            self.results_dict['learning_actions'])
        self.results_dict['corrections'] = np.hstack(
            self.results_dict['corrections'])
        self.results_dict['kinf'] = np.vstack(self.results_dict['kinf'])

    def _setup_constraints(self):
        r"""Set up constraints."""

        # Tightened boundaries for model
        self.tightened_state_space = deepcopy(self.env.observation_space)
        self.tightened_state_space.high -= self.omega
        self.tightened_state_space.low += self.omega
        loggers.info(f'Reduced State Space '
                     f'\n\tfrom: \t{self.env.observation_space} '
                     f'\n\tto: \t{self.tightened_state_space}')

        # \bar U = U - K_\omega \Omega
        ones = np.ones(self.env.observation_space.shape[0])
        K_Omega_omega = self.lqr_gain @ (self.omega * ones)
        # TODO: might change to action -1, +1 range
        #acc = self.env.post_process_action(K_Omega_omega)
        acc = K_Omega_omega
        self.tightened_action_space = deepcopy(self.env.action_space)
        self.tightened_action_space.high += acc
        self.tightened_action_space.low -= acc
        loggers.info(f'Reduced Action Space '
                     f'\n\tfrom: \t{self.env.action_space} '
                     f'\n\tto: \t{self.tightened_action_space}')

        state_con = Munch.fromDict(dict(
            constraint_form='box_bound',
            constrained_variable='STATE',
            upper_bounds=self.tightened_state_space.high,
            lower_bounds=self.tightened_state_space.low,
        ))
        input_con = Munch.fromDict(dict(
            constraint_form='box_bound',
            constrained_variable='INPUT',
            upper_bounds=self.tightened_action_space.high,
            lower_bounds=self.tightened_action_space.low,
        ))
        self.constraint_list = [state_con, input_con]
        AVAILABLE_CONSTRAINTS = {
            "box_bound": BoxConstraint
        }

        self.constraints = create_ConstraintList_from_list(
            self.constraint_list, AVAILABLE_CONSTRAINTS, model=self.model
        )

        # todo: may include omega into constraint list..
        Omega = 0.01  # measured as (x-y).T @ P @ (x-y)

        self.omega_constraint = QuadraticConstraint(
            self.model, P=np.eye(2), b=Omega**2,
            constrained_variable=ConstrainedVariableType.STATE
        )

    def setup_results_dict(self):
        """Setup the results dictionary to store run information.
        """
        self.results_dict = {'obs': [], 'actions': [], 'uncertified_obs': [],
                             'uncertified_actions': [], 'cost': [],
                             'learning_actions': [], 'corrections': [],
                             'feasible': [], 'kinf': []}

    def setup_optimizer(self):
        """Setup the certifying MPC problem.
        """
        loggers.debug('Setup Optimizer...')
        # Horizon parameter.
        horizon = self.horizon
        nx, nu = self.nx, self.nu

        opti = cs.Opti()
        z_var = opti.variable(nx, horizon + 1)
        v_var = opti.variable(nu, horizon)
        u_tilde = opti.variable(nu, 1)
        u_L = opti.parameter(nu, 1)
        x = opti.parameter(nx, 1)

        # === Constraints
        # currently supports only a single constraint for state and input
        con_state = self.constraints.state_constraints[0]
        con_input = self.constraints.input_constraints[0]
        state_constraints = con_state.get_symbolic_model()
        input_constraints = con_input.get_symbolic_model()
        omega_constraint = self.omega_constraint.get_symbolic_model()

        for i in range(self.horizon):
            # Dynamics constraints (eqn 5.b).
            next_state = self.dynamics_nn_func(
                z_var[:, i], v_var[:, i], *self.nn_parameters
            )
            opti.subject_to(z_var[:, i + 1] == next_state)
            # Input constraints (eqn 5.c).
            opti.subject_to(input_constraints(v_var[:, i]) <= 0)
            # State Constraints
            opti.subject_to(state_constraints(z_var[:, i]) <= 0)

        opti.subject_to(omega_constraint(z_var[:, -1]) <= 0)
        # <<<<<<<

        # Initial state constraints (5.e).
        opti.subject_to(omega_constraint(x - z_var[:, 0]) <= 0)
        # Real input (5.f).
        # todo: self.lqr_gain @ (x - z_var[:, 0])
        opti.subject_to(u_tilde == (v_var[:, 0] + x - z_var[:, 0]))
        # Cost (# eqn 5.a, note: using 2norm or sqrt makes this infeasible).
        cost = (u_L - u_tilde).T @ (u_L - u_tilde)
        opti.minimize(cost)
        # Create solver (IPOPT solver as of this version).
        opts = {
            "ipopt.print_level": 0,  # change to 5 for detailed console prints
            "ipopt.sb": "yes",
            "ipopt.max_iter": 50,
            "print_time": 0  # prints the time taken for IP-Opt
        }
        opti.solver('ipopt', opts)
        self.opti_dict = {
            "opti": opti,
            "z_var": z_var,
            "v_var": v_var,
            "u_tilde": u_tilde,
            "u_L": u_L,
            "x": x,
            "cost": cost
        }
        loggers.info('Done setup optimizer.')

    def solve_optimization(self,
                           obs: np.ndarray,
                           uncertified_input: np.ndarray,
                           target_state: Optional[np.ndarray] = None,
                           ):
        """Solve the MPC optimization problem for a given observation and uncertified input.
        """
        loggers.debug(f'Solve_optimization with x: {obs} target state: {target_state}'
                      f'u: {uncertified_input} ')
        opti_dict = self.opti_dict
        opti = opti_dict["opti"]
        z_var = opti_dict["z_var"]
        v_var = opti_dict["v_var"]
        u_tilde = opti_dict["u_tilde"]
        u_L = opti_dict["u_L"]
        x = opti_dict["x"]
        cost = opti_dict["cost"]

        opti.set_value(u_L, uncertified_input)
        if target_state is not None:
            # Tracking problem: Transform observation into error space
            loggers.debug(f'Use target: {target_state}')
            loggers.debug(f'Use error: {obs - target_state}')
            opti.set_value(x, obs - target_state)
        else:
            opti.set_value(x, obs)

        # Initial guess for optimization problem.
        if (self.warm_start and
                self.z_prev is not None and
                self.v_prev is not None and
                self.u_tilde_prev is not None):
            # Shift previous solutions by 1 step.
            z_guess = deepcopy(self.z_prev)
            v_guess = deepcopy(self.v_prev)
            z_guess[:, :-1] = z_guess[:, 1:]
            v_guess[:, :-1] = v_guess[:, 1:]
            # print(f'Re-use values:', z_guess.T)
            opti.set_initial(z_var, z_guess)
            opti.set_initial(v_var, v_guess)
            opti.set_initial(u_tilde, deepcopy(self.u_tilde_prev))
        # Solve the optimization problem.
        try:
            sol = opti.solve()
            z_val, v_val, u_tilde_val = sol.value(z_var), sol.value(
                v_var), sol.value(u_tilde)
            self.z_prev = z_val
            self.v_prev = v_val.reshape((self.model.nu, -1))
            self.u_tilde_prev = u_tilde_val

            # Take the first one from solved action sequence.
            if v_val.ndim > 1:
                action = u_tilde_val
            else:
                action = u_tilde_val
            self.prev_action = u_tilde_val
            feasible = True

            # z_0 = z_var[:, 0]
            # dist = (z_0 - obs).T @ (z_0 - obs)
            # print(f'(z_0-x).T P (z_0-x): {dist}')
        except RuntimeError:
            feasible = False
            action = None
        return action, feasible

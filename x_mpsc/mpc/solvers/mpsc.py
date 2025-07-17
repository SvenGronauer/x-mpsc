r"""Solver classes and functions for safe reinforcement learning.

"""
import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
import casadi as cs
from copy import deepcopy
from scipy import linalg
from munch import Munch

# local imports:
import x_mpsc.common.loggers as loggers
import x_mpsc.mpc.core as mpc_core
import x_mpsc.models
from x_mpsc.mpc.constraints import create_ConstraintList_from_list, \
    BoxConstraint, QuadraticConstraint, ConstrainedVariableType, \
    LinearConstraint


class MPSC(mpc_core.Solver):
    r"""Model Predictive Safety Certification Solver (MPSC)."""
    def __init__(self,
                 env: gym.Env,
                 model: x_mpsc.models.SymbolicModel,  #: SymbolicModel,
                 omega: float,
                 horizon: int = 10,
                 debug: bool = False
                 ):
        # Call super class
        super(MPSC, self).__init__(env, model, debug)

        np.set_printoptions(precision=3)

        self.omega = omega
        # === Sanity checks
        # must be skipped when nominal model is different from real system

        # assert np.allclose(self.env.A, self.model.discrete_dfdx)
        # loggers.debug('A == dfdx? Passed')
        # print(self.env.B - self.model.discrete_dfdu)
        # assert np.allclose(self.env.B, self.model.discrete_dfdu, atol=1.e-3)
        # loggers.debug('B == dfdu? Passed')

        self.z_prev = None
        self.v_prev = None
        self.u_tilde_prev = None
        self.prev_action = None

        self.horizon = horizon

        self.results_dict = {}

        self.kinf = 0
        self.time_step = 0

        # === Linearize system dynamics
        # self.linear_dynamics_func = self.model.get_linear_dynamics_function()
        # self.X_LIN = np.atleast_2d(self.env.X_GOAL)[0, :].T
        # self.U_LIN = np.atleast_2d(self.env.U_GOAL)[0, :]
        #
        # dfdxdfdu = self.model.df_func(x=self.X_LIN, u=self.U_LIN)
        # dfdx = dfdxdfdu['dfdx'].toarray()
        # dfdu = dfdxdfdu['dfdu'].toarray()
        #
        # self.discrete_dfdx, self.discrete_dfdu = discretize_linear_system(
        #     dfdx, dfdu, self.env.dt)

        self.linear_dynamics_func = self.model.linear_dynamics_func
        self.discrete_dfdx = self.model.discrete_dfdx
        self.discrete_dfdu = self.model.discrete_dfdu

        print(f'MPSC linearization:\n{self.discrete_dfdx}')

        self.lqr_gain = self.compute_lqr_gain()

        # Setup constraints based on observation and action space of env
        self.setup_constraints()

        # Now that constraints are defined, setup the optimizer.
        self.setup_optimizer()

        self.results_dict = {}
        self.setup_results_dict()

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
            self.kinf = 0
        else:
            self.kinf += 1
            if (self.kinf <= self.horizon - 1 and
                    self.z_prev is not None and
                    self.v_prev is not None):
                # clip kinf (in case we use a horizon update scheme
                if self.kinf >= self.v_prev.shape[1]:
                    self.kinf = self.v_prev.shape[1] - 1

                err = obs - target_state - self.z_prev[:, self.kinf]
                
                loggers.trace(f'-- z_prev.shape: {self.z_prev.shape}')
                loggers.trace(f'-- Error: {err}')
                loggers.trace(f'-- Adjust: {self.lqr_gain @ err}')
                loggers.trace(f'-- v_prev.shape: {self.v_prev.shape}')
                loggers.trace(f'-- v_prev: {self.v_prev[:, self.kinf]}')
                action = self.v_prev[:, self.kinf] + self.lqr_gain @ err
                loggers.trace(f'-- action: {action}')
            else:
                action = self.lqr_gain @ obs

        action = np.asarray(action).reshape(u_L.shape)
        self.results_dict['kinf'].append(self.kinf)

        action_diff = np.linalg.norm(u_L - action)
        loggers.debug(f'final action: {action}')
        self.results_dict['actions'].append(action)
        self.results_dict['corrections'].append(action_diff)

        return action, feasible

    # Note: moved to SymbolicModel class
    def compute_lqr_gain(self):
        """Compute LQR gain by solving the DARE.

        """
        Q, R = self.env.Q, self.env.R
        df_dx, df_du = self.discrete_dfdx, self.discrete_dfdu
        P = linalg.solve_discrete_are(df_dx, df_du, Q, R)
        btp = np.dot(df_du.T, P)
        lqr_gain = -np.dot(np.linalg.inv(R + np.dot(btp, df_du)), np.dot(btp, df_dx))
        return lqr_gain

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

    def setup_constraints(self):
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
        acc =K_Omega_omega
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
        nx, nu = self.model.nx, self.model.nu
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        z_var = opti.variable(nx, horizon + 1)
        # Inputs.
        v_var = opti.variable(nu, horizon)
        # Certified input.
        u_tilde = opti.variable(nu, 1)
        # Desired input.
        u_L = opti.parameter(nu, 1)
        # Current observed state.
        x = opti.parameter(nx, 1)

        # === Constraints
        # currently supports only a single constraint for state and input
        con_state = self.constraints.state_constraints[0]
        con_input = self.constraints.input_constraints[0]
        state_constraints = con_state.get_symbolic_model()
        input_constraints = con_input.get_symbolic_model()
        # todo: make omega_constraint more generic?
        omega_constraint = self.omega_constraint.get_symbolic_model()

        # print(f'self.model.constraints.state_constraints: {self.model.constraints.state_constraints}')
        # if len(self.constraints.state_constraints) > 1:
        #     con_state2 = self.constraints.state_constraints[1]
        #     state_constraints2 = con_state2.get_symbolic_model()

        for i in range(self.horizon):
            # Dynamics constraints (eqn 5.b).
            next_state = \
            self.linear_dynamics_func(x0=z_var[:, i], p=v_var[:, i])['xf']
            opti.subject_to(z_var[:, i + 1] == next_state)
            # Input constraints (eqn 5.c).
            opti.subject_to(input_constraints(v_var[:, i]) <= 0)
            # State Constraints
            opti.subject_to(state_constraints(z_var[:, i]) <= 0)

            # if len(self.constraints.state_constraints) > 1:
            #     opti.subject_to(state_constraints2(z_var[:, i]) <= 0)
            #     loggers.info('ADDED 2. Constraint!!!')

        # Note: Removed Eq. 5d
        # # Final state constraints (5.d).
        # opti.subject_to(z_var[:, -1] == 0)
        # todo: new: Set final omega
        opti.subject_to(omega_constraint(z_var[:, -1]) <= 0)
        # <<<<<<<

        # Initial state constraints (5.e).
        opti.subject_to(omega_constraint(x - z_var[:, 0]) <= 0)
        # Real input (5.f).
        opti.subject_to(
            u_tilde == v_var[:, 0] + self.lqr_gain @ (x - z_var[:, 0]))
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

r"""Core classes and functions for safe reinforcement learning.

"""
import numpy as np
from typing import Optional
import casadi as cs
import scipy
from scipy import linalg

# local imports:
import x_mpsc.common.loggers as loggers
from x_mpsc.models.base import DynamicsModel


class SymbolicModel(DynamicsModel):
    """Implements the dynamics model with symbolic variables.
    x_dot = f(x,u), y = g(x,u), with other pre-defined, symbolic functions
    (e.g. cost, constraints), serve as priors for the controllers.
    Notes:
        * naming convention on symbolic variable and functions.
            * for single-letter symbol, use {}_sym, otherwise use underscore for delimiter.
            * for symbolic functions to be exposed, use {}_func.

    Source:
    https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/math_and_models/symbolic_systems.py
    """

    def __init__(self,
                 cost: dict,
                 dt: float,
                 dynamics: dict,
                 integration_algo: str = 'cvodes',
                 funcs: Optional[dict] = None,
                 ):
        """
        """
        self.dt = dt  # Sampling time in [s]

        # setup dynamics
        self.x_sym = dynamics["vars"]["X"]
        self.u_sym = dynamics["vars"]["U"]
        self.x_dot = dynamics["dyn_eqn"]
        if dynamics["obs_eqn"] is None:
            self.y_sym = self.x_sym
        else:
            self.y_sym = dynamics["obs_eqn"]

        super().__init__(
            nx=self.x_sym.shape[0],
            nu=self.u_sym.shape[0],
            ny=self.y_sym.shape[0]
        )

        # Integration algorithm.
        self.integration_algo = integration_algo
        # Other symbolic functions.
        if funcs is not None:
            for name, func in funcs.items():
                assert name not in self.__dict__
                self.__dict__[name] = func
        # Variable dimensions.
        self.nx = self.x_sym.shape[0]
        self.nu = self.u_sym.shape[0]
        self.ny = self.y_sym.shape[0]
        # Setup cost function.
        self.cost_func = cost["cost_func"]
        print(self.cost_func)
        self.Q = cost["vars"]["Q"]
        self.R = cost["vars"]["R"]

        self.X_GOAL = np.zeros(2)
        self.U_GOAL = np.zeros(1)

        # Setup symbolic model.
        self._setup_model()

        # Setup Jacobian and Hessian of the dynamics and cost functions.
        self._setup_linearization()

        # === Linearize system dynamics
        self.linear_dynamics_func = self.get_linear_dynamics_function()
        self.X_LIN = np.atleast_2d(self.X_GOAL)[0, :].T
        self.U_LIN = np.atleast_2d(self.U_GOAL)[0, :]

        dfdxdfdu = self.df_func(x=self.X_LIN, u=self.U_LIN)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()

        self.discrete_dfdx, self.discrete_dfdu = self.discretize_linear_system(
            dfdx, dfdu, self.dt)

        loggers.info(f'discrete_dfdx:\n{self.discrete_dfdx}')
        loggers.info(f'discrete_dfdu:\n{self.discrete_dfdu}')

        # self.lqr_gain = self.compute_lqr_gain()
        # loggers.info(f'Computed K_omega: \n{self.lqr_gain}')

        # self.constraints = None
    #     self._setup_constraints()
    #
    # @abc.abstractmethod
    # def _setup_constraints(self,
    #                        action_space: Optional[gym.Space] = None,
    #                        observation_space: Optional[gym.Space] = None
    #                        ) -> None:
    #     pass

    def _setup_model(self):
        """Exposes functions to evaluate the model.
        """
        # Continuous time dynamics.
        self.fc_func = cs.Function(
            'fc', [self.x_sym, self.u_sym], [self.x_dot], ['x', 'u'], ['f']
        )
        # Discrete time dynamics.
        self.fd_func = cs.integrator(
            'fd', self.integration_algo,
            {'x': self.x_sym, 'p': self.u_sym, 'ode': self.x_dot},
            {'tf': self.dt}
        )
        # Observation model.
        self.g_func = cs.Function(
            'g', [self.x_sym, self.u_sym], [self.y_sym], ['x', 'u'], ['g']
        )

    def _setup_linearization(self):
        """Exposes functions for the linearized model.
        """
        # Jacobians w.r.t state & input.
        self.dfdx = cs.jacobian(self.x_dot, self.x_sym)
        self.dfdu = cs.jacobian(self.x_dot, self.u_sym)
        self.df_func = cs.Function('df', [self.x_sym, self.u_sym],
                                   [self.dfdx, self.dfdu], ['x', 'u'],
                                   ['dfdx', 'dfdu'])
        self.dgdx = cs.jacobian(self.y_sym, self.x_sym)
        self.dgdu = cs.jacobian(self.y_sym, self.u_sym)
        self.dg_func = cs.Function('dg', [self.x_sym, self.u_sym],
                                   [self.dgdx, self.dgdu], ['x', 'u'],
                                   ['dgdx', 'dgdu'])
        # Evaluation point for linearization.
        self.x_eval = cs.MX.sym('x_eval', self.nx, 1)
        self.u_eval = cs.MX.sym('u_eval', self.nu, 1)
        # Linearized dynamics model.
        self.x_dot_linear = self.x_dot + self.dfdx @ (
            self.x_eval - self.x_sym) + self.dfdu @ (self.u_eval - self.u_sym)
        self.fc_linear_func = cs.Function(
            'fc', [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.x_dot_linear], ['x_eval', 'u_eval', 'x', 'u'], ['f_linear'])
        self.fd_linear_func = cs.integrator(
            'fd_linear', self.integration_algo, {
                'x': self.x_eval,
                'p': cs.vertcat(self.u_eval, self.x_sym, self.u_sym),
                'ode': self.x_dot_linear
            }, {'tf': self.dt})
        # Linearized observation model.
        self.y_linear = self.y_sym + self.dgdx @ (
            self.x_eval - self.x_sym) + self.dgdu @ (self.u_eval - self.u_sym)
        self.g_linear_func = cs.Function(
            'g_linear', [self.x_eval, self.u_eval, self.x_sym, self.u_sym],
            [self.y_linear], ['x_eval', 'u_eval', 'x', 'u'], ['g_linear'])
        # Jacobian and Hessian of cost function.
        self.l_x = cs.jacobian(self.cost_func, self.x_sym)
        self.l_xx = cs.jacobian(self.l_x, self.x_sym)
        self.l_u = cs.jacobian(self.cost_func, self.u_sym)
        self.l_uu = cs.jacobian(self.l_u, self.u_sym)
        self.l_xu = cs.jacobian(self.l_x, self.u_sym)
        l_inputs = [self.x_sym, self.u_sym, # self.Xr, self.Ur,
                    self.Q, self.R]
        l_inputs_str = ['x', 'u', # 'Xr', 'Ur',
                        'Q', 'R']
        l_outputs = [self.cost_func, self.l_x, self.l_xx, self.l_u, self.l_uu, self.l_xu]
        l_outputs_str = ['l', 'l_x', 'l_xx', 'l_u', 'l_uu', 'l_xu']
        self.loss = cs.Function('loss', l_inputs, l_outputs, l_inputs_str, l_outputs_str)

    # def compute_lqr_gain(self):
    #     """Compute LQR gain by solving the DARE.
    #
    #     """
    #     Q, R = self.env.Q, self.env.R
    #     df_dx, df_du = self.discrete_dfdx, self.discrete_dfdu
    #     P = linalg.solve_discrete_are(df_dx, df_du, Q, R)
    #     btp = np.dot(df_du.T, P)
    #     lqr_gain = -np.dot(np.linalg.inv(R + np.dot(btp, df_du)), np.dot(btp, df_dx))
    #     return lqr_gain

    @classmethod
    def discretize_linear_system(
            cls,
            A,
            B,
            dt,
            exact=False
    ):
        """Discretize a linear system.

        dx/dt = A x + B u
        --> xd[k+1] = Ad xd[k] + Bd ud[k] where xd[k] = x(k*dt)

        Args:
            A: np.array, system transition matrix.
            B: np.array, input matrix.
            dt: scalar, step time interval.
            exact: bool, if to use exact discretization.

        Returns:
            Discretized matrices Ad, Bd.

        """
        state_dim, input_dim = A.shape[1], B.shape[1]
        if exact:
            M = np.zeros((state_dim + input_dim, state_dim + input_dim))
            M[:state_dim, :state_dim] = A
            M[:state_dim, state_dim:] = B
            Md = scipy.linalg.expm(M * dt)
            Ad = Md[:state_dim, :state_dim]
            Bd = Md[:state_dim, state_dim:]
        else:
            I = np.eye(state_dim)
            Ad = I + A * dt
            Bd = B * dt
        return Ad, Bd

    def get_linear_dynamics_function(self) -> cs.Function:
        dfdxdfdu = self.df_func(x=self.X_GOAL, u=self.U_GOAL)
        dfdx = dfdxdfdu['dfdx'].toarray()
        dfdu = dfdxdfdu['dfdu'].toarray()
        delta_x = cs.MX.sym('delta_x', self.nx, 1)
        delta_u = cs.MX.sym('delta_u', self.nu, 1)
        x_dot_lin_vec = dfdx @ delta_x + dfdu @ delta_u
        linear_dynamics_func = cs.integrator(
            'linear_discrete_dynamics', self.integration_algo,
            {
                'x': delta_x,
                'p': delta_u,
                'ode': x_dot_lin_vec
            }, {'tf': self.dt}
        )
        # discrete_dfdx, discrete_dfdu = \
        #     discretize_linear_system(dfdx, dfdu, self.dt)
        return linear_dynamics_func



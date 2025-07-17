r"""Symbolic dynamic model for mass-spring-damper system.

"""
import numpy as np
import casadi as cs

# local
import x_mpsc.common.loggers as loggers
from x_mpsc.models import SymbolicModel


class MassSpringDamperModel(SymbolicModel):
    def __init__(
            self,
            dt: float = 0.1,
            omega: float = 0.1,
    ):
        self.omega = omega
        self.dt = dt  # [s]
        spring_damp = 0.5
        vel_damp = 0.1

        # Input variables.
        x = cs.MX.sym('x')  # position
        x_dot = cs.MX.sym('x_dot')
        self.X = X = cs.vertcat(x, x_dot)  # [pos, vel]^T
        self.U = U = cs.MX.sym('U')
        self.nx = nx = 2
        self.nu = nu = 1

        self.X_GOAL = np.zeros(2)
        self.U_GOAL = np.zeros(1)

        # Dynamics. next state via dynamics:  X_dot = A @ self.X + B @ self.U
        X_dot = cs.vertcat(x_dot, U - spring_damp * x - vel_damp * x_dot)

        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)

        self.cost_func = 0.5 * X.T @ Q @ X + 0.5 * U.T @ R @ U

        # Define dynamics and cost dictionaries.
        # note if "obs_eqn": None then obs_eqn = X
        dynamics = {"dyn_eqn": X_dot, "obs_eqn": None, "vars": {"X": X, "U": U}}
        cost = {"cost_func": self.cost_func,
                "vars": {"X": X, "U": U,  # "Xr": Xr, "Ur": Ur,
                         "Q": Q, "R": R}}

        # Setup symbolic model.
        super(MassSpringDamperModel, self).__init__(
            cost=cost,
            dt=dt,
            dynamics=dynamics,
        )
        loggers.info('Setup symbolic model of: Mass Spring Damper')


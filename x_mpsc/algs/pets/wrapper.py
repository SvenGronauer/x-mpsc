from typing import Union, Tuple
import torch as th
import torch.nn as nn


import casadi as cs
import sys
sys.path.append("../pets_torch")

# local imports
from x_mpsc.mpsc.global_variables import MAX_VARIANCE, MIN_VARIANCE

import x_mpsc.common.loggers as loggers
from x_mpsc.common.utils import to_tensor, to_matrix
from nn_models.prob_ensemble import ProbEnsemble
import numpy as np


class MLP(nn.Module):
    def __init__(
            self,
            act_dim: int = 1,
            obs_dim: int = 2,
            hidden1: int = 200,
            hidden2: int = 200,
            out_dim: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+act_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, out_dim),
            )

    def set_parameters(self, w0, b0, w1, b1, w2, b2):
        with th.no_grad():
            self.net[0].weight = nn.Parameter(to_tensor(w0))
            self.net[0].bias = nn.Parameter(to_tensor(b0.flatten()))

            self.net[2].weight = nn.Parameter(to_tensor(w1))
            self.net[2].bias = nn.Parameter(to_tensor(b1.flatten()))

            self.net[4].weight = nn.Parameter(to_tensor(w2))
            self.net[4].bias = nn.Parameter(to_tensor(b2.flatten()))

    def forward(
            self,
            x: th.Tensor,
            u: th.Tensor,
    ):
        z = th.cat((x, u), dim=-1)
        y = self.net(z)
        return y


class CasadiProbEnsembleWrapper(object):
    def __init__(
            self,
            model: ProbEnsemble,
            nx: int,
            nu: int,
            model_idx: int
    ):
        self.model = model
        self.M = self.model.ens_size

        self.nx = nx
        self.nu = nu

        self.x_sym = cs.MX.sym('x', nx, 1)
        self.u_sym = cs.MX.sym('u', nu, 1)

        # === Mean
        self.w0 = model.fc_layers[0].lin_w[model_idx].cpu().detach().numpy().T
        self.b0 = model.fc_layers[0].lin_b[model_idx].cpu().detach().numpy().T

        self.w1 = model.fc_layers[1].lin_w[model_idx].cpu().detach().numpy().T
        self.b1 = model.fc_layers[1].lin_b[model_idx].cpu().detach().numpy().T

        self.w2 = model.fc_layers[2].lin_w[model_idx].cpu().detach().numpy().T
        self.b2 = model.fc_layers[2].lin_b[model_idx].cpu().detach().numpy().T

        self.w0_sym = cs.MX.sym('w0', *self.w0.shape)
        self.b0_sym = cs.MX.sym('b0', *self.b0.shape)
        self.w1_sym = cs.MX.sym('w1', *self.w1.shape)
        self.b1_sym = cs.MX.sym('b1', *self.b1.shape)
        self.w2_sym = cs.MX.sym('w2', *self.w2.shape)
        self.b2_sym = cs.MX.sym('b2', *self.b2.shape)


        input_shape = (nx+nu, 1)
        self.inputs_mean = np.zeros(input_shape)
        self.inputs_sigma = np.ones(input_shape)
        if hasattr(model, "inputs_mu"):
            self.inputs_mean = model.inputs_mu.data.numpy().reshape(input_shape)
            self.inputs_sigma = model.inputs_sigma.data.numpy().reshape(input_shape)

        self.in_mu_sym = cs.MX.sym('in_mu', *self.inputs_mean.shape)
        self.in_sigma_sym = cs.MX.sym('in_sigma', *self.inputs_sigma.shape)

        self.output_vec = cs.mtimes(self.w2_sym, cs.tanh(cs.mtimes(self.w1_sym, cs.tanh(cs.mtimes(self.w0_sym, (cs.vertcat(self.x_sym, self.u_sym) - self.in_mu_sym) / self.in_sigma_sym) + self.b0_sym)) + self.b1_sym)) + self.b2_sym

        self.f_discrete_func = cs.Function(
            'f_discrete_func',
            [self.x_sym, self.u_sym, self.w0_sym, self.b0_sym,
             self.w1_sym, self.b1_sym, self.w2_sym, self.b2_sym,
             self.in_mu_sym, self.in_sigma_sym],
            [self.output_vec, ]
        )

        self.torch_net = MLP(nu, nx, self.w0.shape[0], self.w1.shape[0], 2*nx)
        self.torch_net.set_parameters(self.w0, self.b0, self.w1,
                                      self.b1, self.w2, self.b2)

        # todo sven: implement swish function
        # self.swish = cs.Function(
        #     'swish_activation',
        # )

        self.df_dx = cs.jacobian(self.output_vec, self.x_sym)
        self.df_du = cs.jacobian(self.output_vec, self.u_sym)

        self.df_dx_func = cs.Function(
            'df_dx_func',
            [self.x_sym, self.u_sym, self.w0_sym, self.b0_sym,
             self.w1_sym, self.b1_sym, self.w2_sym, self.b2_sym,
             self.in_mu_sym, self.in_sigma_sym],
            [self.df_dx, ]
        )
        self.df_du_func = cs.Function(
            'df_du_func',
            [self.x_sym, self.u_sym, self.w0_sym, self.b0_sym,
             self.w1_sym, self.b1_sym, self.w2_sym, self.b2_sym,
             self.in_mu_sym, self.in_sigma_sym],
            [self.df_du, ]
        )


        self.check_parameters()
        loggers.debug('Network successfully transformed into casadi function.')

    #todo sven: this method can be removed in future
    def compute_torch_jacobians(
            self,
            x: np.ndarray,
            u: np.array
    ):
        z = th.cat((to_tensor(x), to_tensor(u)), dim=-1)
        z_matrix = th.stack([z for _ in range(5)])
        net = self.model.fc_layers
        # out = self.model.fc_layers(z).detach().cpu().numpy()
        jac = th.autograd.functional.jacobian(net.forward, z_matrix)
        return jac.numpy()

    def check_parameters(self):
        BATCH_SIZE = 2
        # z_np = (np.ones((self.nx+self.nu, 1)) - self.inputs_mean) / self.inputs_sigma
        # z_th = th.as_tensor(np.ones((self.M, BATCH_SIZE, self.nx+self.nu)), dtype=th.float32)
        #
        # with th.no_grad():
        #     casadi_model_outputs = np.array([self.forward_numpy(x, u)])
        #     casadi_model_outputs = np.swapaxes(casadi_model_outputs, 1, 2)
        #     torch_model_output = self.model.fc_layers(z).detach().cpu().numpy()

        # assert np.allclose(casadi_model_outputs, torch_model_output)

        """--- new --- vector inputs"""
        x = np.array([np.pi, 0.2], dtype=np.float32)
        u = np.ones(self.nu)
        with th.no_grad():
            x_next_torch_model = self.torch_net(to_tensor(x), to_tensor(u)).numpy()
        x_next_casadi_model = self.forward_numpy(x, u)
        np.allclose(x_next_torch_model, x_next_casadi_model)

    @property
    def params(self):
        return [self.w0, self.b0, self.w1, self.b1,
                self.w2, self.b2, self.inputs_mean, self.inputs_sigma]

    def forward_numpy(self,
                      x: np.ndarray,
                      u: np.ndarray
                      ) -> np.ndarray:
        output = self.f_discrete_func(to_matrix(x), to_matrix(u), *self.params).full()
        return output

    def f_discrete(
            self,
            x: Union[cs.DM, cs.MX, cs.SX],
            u: Union[cs.DM, cs.MX, cs.SX]
    ):
        return self.f_discrete_func(x, u, *self.params)[:self.nx]

    def get_df_du(
            self,
            x: Union[cs.DM, cs.MX, cs.SX],
            u: Union[cs.DM, cs.MX, cs.SX]
    ):
        r"""Returns the Jacobian with respect to x."""
        return self.df_du_func(x, u, *self.params)[:self.nx, :self.nu]

    def get_df_dx(
            self,
            x: Union[cs.DM, cs.MX, cs.SX],
            u: Union[cs.DM, cs.MX, cs.SX]
    ):
        r"""Returns the Jacobian with respect to x."""
        # print(self.df_dx_func(x, u, *self.params)[:, :])
        return self.df_dx_func(x, u, *self.params)[:self.nx, :self.nx]

    def get_Q(
            self,
            x: Union[cs.DM, cs.MX, cs.SX],
            u: Union[cs.DM, cs.MX, cs.SX]
    ) -> Union[cs.DM, cs.MX, cs.SX]:
        r"""Returns the shape matrix of the uncertainty ellipsoid."""
        y = self.f_discrete_func(x, u, *self.params)
        variance = cs.exp(y[self.nx:])
        clipped_variance = cs.fmin(cs.fmax(variance, MIN_VARIANCE), MAX_VARIANCE)
        return cs.diag(clipped_variance)

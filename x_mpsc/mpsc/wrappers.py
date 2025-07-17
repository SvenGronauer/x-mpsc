from __future__ import annotations
from typing import Union, Tuple, Optional
import numpy as np
import torch as th
import torch.nn as nn
import casadi as cs

# local imports
from x_mpsc.mpsc.global_variables import MAX_VARIANCE, MIN_VARIANCE
import x_mpsc.common.loggers as loggers
from x_mpsc.common.utils import to_tensor, to_matrix


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
        in_features = obs_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, out_dim),
            )
        self.inputs_mu = nn.Parameter(th.zeros(1, in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(th.ones(1, in_features), requires_grad=False)

    def set_parameters(self, w0, b0, w1, b1, w2, b2, inputs_mu, inputs_std):
        with th.no_grad():
            self.inputs_mu = nn.Parameter(to_tensor(inputs_mu.flatten()))
            self.inputs_std = nn.Parameter(to_tensor(inputs_std.flatten()))

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
        z_std = (z - self.inputs_mu) / (self.inputs_sigma + 1e-10)
        y = self.net(z_std)
        return y


class EnsembleModelCasadiWrapper:
    def __init__(
            self,
            dynamics_model, #: DynamicsModel,
            model_idx: int
    ):
        self.model = model = dynamics_model.ensemble_model
        self.M = self.model.ensemble_size
        self.model_idx = model_idx

        self.x_sym = dynamics_model.x_sym
        self.u_sym = dynamics_model.u_sym

        self.prior_dynamics_model = dynamics_model.prior_dynamics_model

        # === Mean
        self.w0 = model.fc_layers[0].lin_w[model_idx].cpu().detach().numpy().T
        self.b0 = model.fc_layers[0].lin_b[model_idx].cpu().detach().numpy().T

        self.w1 = model.fc_layers[2].lin_w[model_idx].cpu().detach().numpy().T
        self.b1 = model.fc_layers[2].lin_b[model_idx].cpu().detach().numpy().T

        self.w2 = model.fc_layers[4].lin_w[model_idx].cpu().detach().numpy().T
        self.b2 = model.fc_layers[4].lin_b[model_idx].cpu().detach().numpy().T

        self.w0_sym = cs.MX.sym('w0', *self.w0.shape)
        self.b0_sym = cs.MX.sym('b0', *self.b0.shape)
        self.w1_sym = cs.MX.sym('w1', *self.w1.shape)
        self.b1_sym = cs.MX.sym('b1', *self.b1.shape)
        self.w2_sym = cs.MX.sym('w2', *self.w2.shape)
        self.b2_sym = cs.MX.sym('b2', *self.b2.shape)

        self.nx = nx = self.x_sym.shape[0]
        self.nu = nu = self.u_sym.shape[0]
        input_shape = (nx + nu, 1)
        self.inputs_mean = np.zeros(input_shape)
        self.inputs_sigma = np.ones(input_shape)
        if hasattr(model, "inputs_mu"):
            self.inputs_mean = model.inputs_mu.data.cpu().numpy().reshape(input_shape)
            self.inputs_sigma = model.inputs_sigma.data.cpu().numpy().reshape(input_shape)

        self.in_mu_sym = cs.MX.sym('in_mu', *self.inputs_mean.shape)
        self.in_sigma_sym = cs.MX.sym('in_sigma', *self.inputs_sigma.shape)

        self.output_vec = cs.mtimes(self.w2_sym, cs.tanh(cs.mtimes(self.w1_sym, cs.tanh(cs.mtimes(self.w0_sym, (cs.vertcat(self.x_sym, self.u_sym) - self.in_mu_sym) / self.in_sigma_sym) + self.b0_sym)) + self.b1_sym)) + self.b2_sym

        if self.prior_dynamics_model is not None:
            # todo sven: implement me
            f = self.prior_dynamics_model.get_casadi_function()
            self.output_vec += f
            # raise NotImplementedError

        self.f_discrete_func = cs.Function(
            'f_discrete_func',
            [self.x_sym, self.u_sym, self.w0_sym, self.b0_sym,
             self.w1_sym, self.b1_sym, self.w2_sym, self.b2_sym,
             self.in_mu_sym, self.in_sigma_sym],
            [self.output_vec, ]
        )

        self.torch_net = MLP(nu, nx, self.w0.shape[0], self.w1.shape[0], 2*nx)
        self.torch_net.set_parameters(self.w0, self.b0, self.w1,
                                      self.b1, self.w2, self.b2,
                                      self.inputs_mean, self.inputs_sigma)

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

    def check_parameters(self):
        x = np.ones(self.nx)
        u = np.ones(self.nu)
        with th.no_grad():
            x_next_torch_model = self.torch_net(to_tensor(x), to_tensor(u)).cpu().numpy()
            xu = np.concatenate((x, u), axis=-1).reshape((1, -1))
            batched_nu = np.repeat(xu, self.M, axis=0)
            batched_nu = batched_nu.reshape((self.M, 1, -1))
            #print(f"xu: {xu}")
            #print(f"batched_nu: {batched_nu}")
            mean, logvar = self.model.forward(to_tensor(batched_nu))
            original_model = mean.cpu().numpy()[self.model_idx].flatten()
            if self.prior_dynamics_model is not None:
                f = self.prior_dynamics_model.predict_next_state
                next_state_from_prio_model = f(x, u).flatten()
                a = 1
                original_model += next_state_from_prio_model


        x_next_casadi_model = self.forward_numpy(x, u).flatten()[:self.nx]
        # print(f"x_next_torch_model: {x_next_torch_model}")
        #print(f"x_next_casadi_model: {x_next_casadi_model}")
        #print(f"original_model: {original_model}")

        #print(f"Delta: {np.linalg.norm(original_model-x_next_casadi_model)}")
        assert np.allclose(original_model, x_next_casadi_model, atol=1.e-5)

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

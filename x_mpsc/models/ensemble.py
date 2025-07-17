import os.path
from typing import Dict, Tuple
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import casadi as cs

# local imports
import x_mpsc.common.mpi_tools as mpi
from x_mpsc.common.utils import get_file_contents
from x_mpsc.algs.utils import get_device
from x_mpsc.common import loggers
from x_mpsc.envs.simple_pendulum.pendulum import SimplePendulumEnv
from x_mpsc.envs.cartpole import CartPoleEnv
from x_mpsc.envs.drone import DroneEnv
from x_mpsc.envs.twolinkarm import TwoLinkArmEnv
from x_mpsc.mpsc.global_variables import MAX_VARIANCE, MIN_VARIANCE


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class StandardScaler(object):
    def __init__(self):
        self.mu = None

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        assert self.mu is not None, 'call fit() before transform()'
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleLinear):
        input_dim = m.in_features
        truncated_normal_init(m.lin_w, std=1 / (2 * np.sqrt(input_dim)))
        m.lin_b.data.fill_(0.0)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class EnsembleLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleLinear, self).__init__()
        self.device = get_device()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.lin_w = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.lin_b = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.lin_w)
        return torch.add(w_times_x, self.lin_b[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.lin_b is not None
        )


class EnsembleModel(nn.Module):
    def __init__(
            self,
            state_size: int,
            action_size: int,
            ensemble_size: int,
            hidden_sizes: Tuple[int],
            learning_rate=1e-3,
            use_decay=False
    ):
        super().__init__()
        self.device = get_device()
        self.hidden_sizes = hidden_sizes
        self.in_features = state_size + action_size
        self.layer_sizes = (self.in_features,) + hidden_sizes + (2*state_size,)
        self.use_decay = use_decay
        self.ensemble_size = ensemble_size
        self.output_dim = state_size  # + reward_size

        self.fc_layers = self._build_ensemble_network()

        self.inputs_mu = nn.Parameter(torch.zeros(1, self.in_features), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.ones(1, self.in_features), requires_grad=False)

        self.max_logvar = nn.Parameter((np.log(MAX_VARIANCE) * torch.ones((1, self.output_dim)).float()).to(self.device), requires_grad=False)
        self.min_logvar = nn.Parameter((np.log(MIN_VARIANCE) * torch.ones((1, self.output_dim)).float()).to(self.device), requires_grad=False)

        self.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # self.swish = Swish()

        self.to(self.device)

    def _build_ensemble_network(self):
        layers = []
        for j in range(len(self.layer_sizes) - 1):
            is_output_layer = j == (len(self.layer_sizes) - 2)
            neurons_in = self.layer_sizes[j]
            neurons_out = self.layer_sizes[j+1]
            act = nn.Tanh if not is_output_layer else nn.Identity
            layer = EnsembleLinear(neurons_in, neurons_out, self.ensemble_size, weight_decay=0.000025*(j+1))
            layers += [layer, act()]
        return nn.Sequential(*layers)

    def forward(self, x, ret_log_var=False):
        x_std = (x - self.inputs_mu) / (self.inputs_sigma + 1e-10)
        net_out = self.fc_layers(x_std)  # forward pass through layers

        mean = net_out[:, :, :self.output_dim]
        logvar = net_out[:, :, self.output_dim:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleLinear):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
                # print(m.weight.shape)
                # print(m, decay_loss, m.weight_decay)
        return decay_loss

    def loss(self, mean, logvar, labels, inc_var_loss=True):
        """
        mean, logvar: Ensemble_size x N x dim
        labels: N x dim
        """
        assert len(mean.shape) == len(logvar.shape) == len(labels.shape) == 3
        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            # Average over batch and dim, sum over ensembles.
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) * inv_var, dim=-1), dim=-1)
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()
        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)
        if self.use_decay:
            loss += self.get_decay_loss()
        loss.backward()
        if mpi.num_procs() > 1:
            mpi.mpi_avg_grads(self.fc_layers)
        self.optimizer.step()


class DynamicsModel:
    def __init__(self,
                 env: gym.Env,
                 ensemble_size: int,
                 elite_size, 
                 hidden_sizes: Tuple[int, int],
                 use_prior_model: bool = False,
                 use_decay=False
                 ):
        assert elite_size <= ensemble_size, "elite_size must be smaller"

        state_size = int(np.prod(env.observation_space.shape))
        action_size = int(np.prod(env.action_space.shape))

        self.ensemble_size = ensemble_size
        self.device = get_device()
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = nx = state_size
        self.action_size = nu = action_size
        self.elite_model_idxes = [i for i in range(elite_size)]
        self.ensemble_model = EnsembleModel(
            state_size, action_size, ensemble_size, hidden_sizes, use_decay=use_decay)
        # self.scaler = StandardScaler()
        self.use_prior_model = use_prior_model

        self.x_sym = cs.MX.sym('x', nx, 1)
        self.u_sym = cs.MX.sym('u', nu, 1)

        if use_prior_model:
            if isinstance(env.unwrapped, SimplePendulumEnv):
                from x_mpsc.models.pendulum_model import SimplePendulumModel
                self.prior_dynamics_model = SimplePendulumModel(
                    x_sym=self.x_sym,
                    u_sym=self.u_sym,
                    # uncertaity = 0.8
                )
            elif isinstance(env.unwrapped, CartPoleEnv):
                from x_mpsc.models.cartpole_model import CartPoleModel
                self.prior_dynamics_model = CartPoleModel(
                    x_sym=self.x_sym,
                    u_sym=self.u_sym,
                )
            elif isinstance(env.unwrapped, TwoLinkArmEnv):
                from x_mpsc.models.twolinkarm_model import TwoLinkArmModel
                self.prior_dynamics_model = TwoLinkArmModel(
                    x_sym=self.x_sym,
                    u_sym=self.u_sym,
                )
            elif isinstance(env.unwrapped, DroneEnv):
                from x_mpsc.models.drone_model import DroneModel
                self.prior_dynamics_model = DroneModel(
                    x_sym=self.x_sym,
                    u_sym=self.u_sym,
                )
            else:
                raise NotImplementedError
        else:
            self.prior_dynamics_model = None

    def fit_scaler(self, xs: np.ndarray):
        if len(xs) < 1:
            loggers.warn("no data for input scaler")
        else:
            # print(f"xs.shape: {xs.shape}")
            mu = mpi.mpi_avg(np.mean(xs, axis=0, keepdims=True))
            # print(f"mu: {mu}")
            sigma = mpi.mpi_avg(np.std(xs, axis=0, keepdims=True))
            sigma[sigma < 1e-12] = 1.0
            # print(f"sigma: {sigma}")
            # print(self.ensemble_model.inputs_mu)
            mu_view = self.ensemble_model.inputs_mu.data.cpu().numpy()
            sigma_view = self.ensemble_model.inputs_sigma.data.cpu().numpy()
            mu_view[:] = mu
            sigma_view[:] = sigma
            # print(self.ensemble_model.inputs_mu)

    @classmethod
    def load(cls, model_path):
        config_fnp = os.path.join(model_path, "config.json")
        model_fnp = os.path.join(model_path, "torch_save/ensemble_model.pt")
        cfg = get_file_contents(config_fnp)
        env = gym.make(cfg["env_id"])


        state_dict = torch.load(model_fnp, map_location=torch.device('cpu'))
        print(f"Loading keys from {model_fnp}:\n{state_dict.keys()}")
        num_hiddens = state_dict['fc_layers.0.lin_w'].shape[2]
        model = DynamicsModel(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            ensemble_size=cfg.get("ensemble_size", 1),
            elite_size=cfg.get("elite_size", 1),
            hidden_sizes=(num_hiddens, num_hiddens),
            use_decay=True
        )

        model.ensemble_model.load_state_dict(state_dict, strict=True)
        return model

    def train(
            self,
            inputs: np.ndarray,
            labels: np.ndarray,
            batch_size=32,
            holdout_ratio=0.0,
            max_epochs_since_update=5,
            max_epochs=1,
            terminate_early: bool = True,
            disable_progress_bar: bool = False,
    ) -> Dict:
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.ensemble_size)}

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, labels = inputs[permutation], labels[permutation]

        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]

        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(self.device)
        holdout_labels = torch.from_numpy(holdout_labels).float().to(self.device)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self.ensemble_size, 1, 1])
        holdout_labels = holdout_labels[None, :, :].repeat([self.ensemble_size, 1, 1])

        eval_mse_losses = []
        train_mse_delta = []
        count = 0
        num_updates_per_epoch = train_inputs.shape[0] // batch_size
        total_updates = max_epochs * num_updates_per_epoch

        desc = f"INFO: \tTrain ensemble"
        d = not(not disable_progress_bar and mpi.is_root_process())
        with tqdm(total=total_updates, desc=desc, ncols=80, unit='updates', disable=d) as pbar:
            for epoch in range(max_epochs):
                count += 1
                mse_losses = []
                train_idx = np.vstack([np.random.permutation(train_inputs.shape[0]) for _ in range(self.ensemble_size)])

                for start_pos in range(0, train_inputs.shape[0], batch_size):
                    idx = train_idx[:, start_pos: start_pos + batch_size]
                    train_input = torch.from_numpy(train_inputs[idx]).float().to(self.device)
                    train_label = torch.from_numpy(train_labels[idx]).float().to(self.device)
                    losses = []
                    mean, logvar = self.ensemble_model(train_input, ret_log_var=True)
                    loss, mse = self.ensemble_model.loss(mean, logvar, train_label)
                    self.ensemble_model.train(loss)
                    losses.append(loss)
                    mse_losses.append(torch.mean(mse).item())
                train_mse_delta.append(mse_losses[-1] - mse_losses[0])
                with torch.no_grad():
                    holdout_mean, holdout_logvar = self.ensemble_model(holdout_inputs, ret_log_var=True)
                    _, holdout_mse_losses = self.ensemble_model.loss(holdout_mean, holdout_logvar, holdout_labels, inc_var_loss=False)
                    holdout_mse_losses = holdout_mse_losses.detach().cpu().numpy()
                    eval_mse_losses.append(np.mean(holdout_mse_losses))

                    sorted_loss_idx = np.argsort(holdout_mse_losses)
                    self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()
                    break_train = self._save_best(epoch, holdout_mse_losses)
                if terminate_early and mpi.mpi_sum(float(break_train)) > 0:
                    break
                pbar.update(num_updates_per_epoch)
        assert len(mse_losses) > 0, "Number of epochs must be positiv"

        return {'Eval/prediction_mse': np.mean(eval_mse_losses),
                'Train/mse_delta': np.mean(train_mse_delta),
                'Train/epochs': count,
                }

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False

    def predict(self, inputs, batch_size=1024, factored=True):
        ensemble_mean, ensemble_var = [], []
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(inputs[i:min(i + batch_size, inputs.shape[0])]).float().to(self.device)
            b_mean, b_var = self.ensemble_model(input[None, :, :].repeat([self.ensemble_size, 1, 1]), ret_log_var=False)
            ensemble_mean.append(b_mean.detach().cpu().numpy())
            ensemble_var.append(b_var.detach().cpu().numpy())
        ensemble_mean = np.hstack(ensemble_mean)
        ensemble_var = np.hstack(ensemble_var)

        if factored:
            return ensemble_mean, ensemble_var
        else:
            assert False, "Need to transform to numpy"
            mean = torch.mean(ensemble_mean, dim=0)
            var = torch.mean(ensemble_var, dim=0) + torch.mean(torch.square(ensemble_mean - mean[None, :, :]), dim=0)
            return mean, var

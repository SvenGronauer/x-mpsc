import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# local imports
from x_mpsc.algs import core

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        network_sizes = [obs_dim] + list(hidden_sizes)
        self.net = core.build_mlp_network(
            network_sizes, activation, output_activation=activation
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            # use the reparametrization trick for stochastic gradients:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation='identity'):
        super().__init__()
        q_network_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]
        self.q = core.build_mlp_network(
            q_network_sizes, activation, output_activation=output_activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self,
                 observation_space,
                 action_space,
                 ac_kwargs: dict,
                 # hidden_sizes=(256, 256),
                 # activation=nn.ReLU
                 ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(
            obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit, **ac_kwargs["pi"]
        )
        self.q1 = MLPQFunction(obs_dim, act_dim, **ac_kwargs["q"])
        self.q2 = MLPQFunction(obs_dim, act_dim, **ac_kwargs["q"])

    def act(self, obs: torch.Tensor, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()

    def forward(self, obs, deterministic=True) -> tuple:
        """Make the interface of this method equal to ActorCritics of On-Policy algorithms."""
        a = self.act(obs, deterministic=deterministic)

        # same return tuple as: return a.numpy(), v.numpy(), logp_a.numpy()
        return a, None, None
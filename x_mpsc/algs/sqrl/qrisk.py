import numpy as np
import itertools
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# local imports
import x_mpsc.common.mpi_tools as mpi
from x_mpsc.algs.sac.actor_critic import MLPQFunction, SquashedGaussianMLPActor
from x_mpsc.algs.sqrl.utils import hard_update
from x_mpsc.algs.utils import get_device, hard_update, soft_update


class QRiskWrapper(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 ac_kwargs: dict,
                 lr: float,
                 tau_safe: float = 0.0002,
                 gamma_safe: float = 0.65,
                 target_update_interval: int = 1,
                 ):
        super().__init__()
        self.device = get_device()
        self.ac_space = action_space
        self.lr = lr

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.q1 = MLPQFunction(
            obs_dim, act_dim, output_activation='sigmoid', **ac_kwargs["q"]
        ).to(device=self.device)
        self.q2 = MLPQFunction(
            obs_dim, act_dim, output_activation='sigmoid', **ac_kwargs["q"]
        ).to(device=self.device)

        self.q1_target = MLPQFunction(
            obs_dim, act_dim, **ac_kwargs["q"]).to(device=self.device)
        self.q2_target = MLPQFunction(
            obs_dim, act_dim, **ac_kwargs["q"]).to(device=self.device)

        self.q_params = itertools.chain(self.q1.parameters(),
                                        self.q2.parameters())
        self.safety_critic_optim = Adam(self.q_params, lr=lr)

        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)

        self.tau = tau_safe
        self.gamma_safe = gamma_safe
        self.updates = 0
        self.target_update_interval = target_update_interval
        self.torchify = lambda x: th.FloatTensor(x).to(self.device)

    def update_parameters(
            self,
            data: dict,
            policy: SquashedGaussianMLPActor
    ):
        '''
        Trains safety critic Q_risk and model-free recovery policy which performs
        gradient ascent on the safety critic

        Arguments:
            data: data from replay buffer
            policy: Agent's composite policy
        '''

        o, a, r, o2 = data['obs'], data['act'], data['rew'], data['obs2']
        d, c = data['done'], data['con']

        with th.no_grad():
            a2, next_state_log_pi = policy(o2)
            qf1_next_target = self.q1_target(o2, a2)
            qf2_next_target = self.q2_target(o2, a2)
            min_qf_next_target = th.max(qf1_next_target, qf2_next_target)
            next_q_value = c + (1 - d) * self.gamma_safe * min_qf_next_target

        qf1 = self.q1(o, a)
        qf2 = self.q2(o, a)
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        self.safety_critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        mpi.mpi_avg_grads(self.q1)  # average grads across MPI processes
        mpi.mpi_avg_grads(self.q2)  # average grads across MPI processes
        self.safety_critic_optim.step()

        if self.updates % self.target_update_interval == 0:
            soft_update(self.q1_target, self.q1, self.tau)
            soft_update(self.q2_target, self.q2, self.tau)
        self.updates += 1

    def get_value(self, o, a):
        '''
            Arguments:
                states, actions --> list of states and list of corresponding
                actions to get Q_risk values for
            Returns: Q_risk(states, actions)
        '''
        with th.no_grad():
            qf1 = self.q1(o, a)
            qf2 = self.q2(o, a)
            return th.max(qf1, qf2)
    #
    # def __call__(self, states, actions):
    #     return self.safety_critic(states, actions)

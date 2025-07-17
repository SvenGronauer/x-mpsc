""" PyTorch implementation of Proximal Policy Optimization (PPO) Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    10.10.2020
Updated:    15.11.2020
"""
import torch
from x_mpsc.algs.iwpg import iwpg
import x_mpsc.algs.utils as utils


class ProximalPolicyOptimizationAlgorithm(iwpg.IWPGAlgorithm):
    def __init__(
            self,
            alg='ppo',
            clip_ratio: float = 0.2,
            **kwargs
    ):
        super().__init__(alg=alg, **kwargs)
        self.clip_ratio = clip_ratio

    def compute_loss_pi(
            self,
            data: dict,
            lstm_state=None,
    ):
        # PPO Policy loss
        dist, _log_p, lstm_state = self.ac.pi(data['obs'], data['act'], lstm_state, data['dones'])
        ratio = torch.exp(_log_p - data['log_p'])

        clip_adv = data['adv'] * torch.clamp(
            ratio,
            1 - self.clip_ratio,
            1 + self.clip_ratio)
        loss_pi = -(torch.min(ratio * data['adv'], clip_adv)).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info


def get_alg(env_id, **kwargs) -> ProximalPolicyOptimizationAlgorithm:
    return ProximalPolicyOptimizationAlgorithm(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='ppo', env_id=env_id)
    defaults.update(**kwargs)
    alg = ProximalPolicyOptimizationAlgorithm(
        env_id=env_id,
        **defaults
    )

    ac, env = alg.learn()

    return ac, env

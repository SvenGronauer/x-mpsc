""" PyTorch implementation of Trust-region Policy Gradient (TRPO) Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    10.10.2020
Updated:    21.11.2020
            09.11.2022  Adopted to X-MPSC repo
"""

import torch
from x_mpsc.algs.iwpg import iwpg
import x_mpsc.algs.utils as U
from x_mpsc.common import utils
import x_mpsc.common.mpi_tools as mpi_tools
from torch.distributions import Normal


@torch.distributions.kl.register_kl(Normal, Normal)
def kl_normal_normal(p, q):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


class TRPOAlgorithm(iwpg.IWPGAlgorithm):
    def __init__(
            self,
            alg: str = 'trpo',
            cg_damping: float = 0.1,
            cg_iters: int = 10,
            target_kl: float = 0.01,
            **kwargs
    ):
        super().__init__(
            alg=alg,
            target_kl=target_kl,
            **kwargs)
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.target_kl = target_kl
        self.fvp_obs = None
        self.scheduler = None  # disable scheduler if activated by parent class


    def adjust_step_direction(self,
                              step_dir,
                              g_flat,
                              p_dist,
                              data,
                              total_steps: int = 15,
                              decay: float = 0.8
                              ) -> tuple:
        """ TRPO performs line-search until constraint satisfaction."""
        step_frac = 1.0
        _theta_old = U.get_flat_params_from(self.ac.pi.net)
        expected_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            U.set_param_values_to_model(self.ac.pi.net, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi, pi_info = self.compute_loss_pi(data=data)
                # determine KL div between new and old policy
                q_dist, _ = self.ac.pi.dist(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(
                    p_dist, q_dist).mean().item()
            loss_improve = self.loss_pi_before - loss_pi.item()
            # average processes....
            torch_kl = mpi_tools.mpi_avg(torch_kl)
            loss_improve = mpi_tools.mpi_avg(loss_improve)

            self.logger.log("Expected Improvement: %.3f Actual: %.3f" % (
                expected_improve, loss_improve))
            if not torch.isfinite(loss_pi):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self.logger.log('INFO: did not improve improve <0')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                self.logger.log(f'Accept step at i={acceptance_step}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        U.set_param_values_to_model(self.ac.pi.net, _theta_old)

        return step_frac * step_dir, acceptance_step

    def algorithm_specific_logs(self):
        self.logger.log_tabular('Misc/AcceptanceStep')
        self.logger.log_tabular('Misc/Alpha')
        self.logger.log_tabular('Misc/FinalStepNorm')
        self.logger.log_tabular('Misc/gradient_norm')
        self.logger.log_tabular('Misc/xHx')
        self.logger.log_tabular('Misc/H_inv_g')

    def Fvp(self, p):
        """ Build the Hessian-vector product based on an approximation of the
            KL-divergence.

            For details see John Schulman's PhD thesis (pp. 40)
            http://joschu.net/docs/thesis.pdf
        """
        self.ac.pi.net.zero_grad()
        q_dist, lstm_state = self.ac.pi.dist(self.fvp_obs)
        with torch.no_grad():
            p_dist, lstm_state = self.ac.pi.dist(self.fvp_obs)

        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self.ac.pi.net.parameters(),
                                    create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * p).sum()
        grads = torch.autograd.grad(kl_p, self.ac.pi.net.parameters(),
                                    retain_graph=False)
        # contiguous indicating, if the memory is contiguously stored or not
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                       for grad in grads])
        # average --->
        mpi_tools.mpi_avg_torch_tensor(flat_grad_grad_kl)
        return flat_grad_grad_kl + p * self.cg_damping

    def update(self) -> None:
        """Update value and policy networks. Note that the order doesn't matter.

        Returns
        -------
            None
        """
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Update Policy Network
        self.update_policy_net(data)
        # Update Value Function
        self.update_value_net(data=data)
        if self.use_cost_value_function:
            self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def update_policy_net(self, data):
        # Get loss and info values before update
        theta_old = U.get_flat_params_from(self.ac.pi.net)
        self.ac.pi.net.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = mpi_tools.mpi_avg(loss_pi.item())
        loss_v = self.compute_loss_v(data['obs'], data['target_v'])
        self.loss_v_before = mpi_tools.mpi_avg(loss_v.item())
        p_dist, lstm_state = self.ac.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        g_flat = U.get_flat_gradients_from(self.ac.pi.net)

        # flip sign since policy_loss = -(ration * adv)
        g_flat *= -1

        x = U.conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        assert xHx.item() >= 0, 'No negative values'

        # perform descent direction
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()

        # determine step direction and apply SGD step after grads where set
        # TRPO uses custom backtracking line search
        final_step_dir, accept_step = self.adjust_step_direction(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            data=data,
        )
        # update actor network parameters
        new_theta = theta_old + final_step_dir
        U.set_param_values_to_model(self.ac.pi.net, new_theta)

        with torch.no_grad():
            q_dist, _ = self.ac.pi.dist(data['obs'])
            kl = torch.distributions.kl.kl_divergence(p_dist,
                                                      q_dist).mean().item()
            loss_pi, pi_info = self.compute_loss_pi(data=data)

        self.logger.store(**{
            'Values/Adv': data['act'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': kl,
            'PolicyRatio': pi_info['ratio'],
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/AcceptanceStep': accept_step,
            'Misc/Alpha': alpha.item(),
            'Misc/StopIter': 1,
            'Misc/FinalStepNorm': torch.norm(final_step_dir).numpy(),
            'Misc/xHx': xHx.item(),
            'Misc/gradient_norm': torch.norm(g_flat).numpy(),
            'Misc/H_inv_g': x.norm().item(),
        })


def get_alg(env_id, **kwargs) -> TRPOAlgorithm:
    return TRPOAlgorithm(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='trpo', env_id=env_id)
    defaults.update(**kwargs)
    alg = TRPOAlgorithm(
        env_id=env_id,
        **defaults
    )
    ac, env = alg.learn()

    return ac, env

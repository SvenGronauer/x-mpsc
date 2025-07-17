r"""Neural-net-based dynamics model for mass-spring-damper system.

"""

import torch as th
import torch.nn as nn

# local
from x_mpsc.models.base import DynamicsModel


class NeuralMassSpringDamperModel(nn.Module, DynamicsModel):
    def __init__(
            self,
            hidden_neurons: int = 50
    ):
        nn.Module.__init__(self)
        DynamicsModel.__init__(self, nx=2, nu=1, ny=2)

        combined_obs_act_dim = self.nx + self.nu
        self.net = nn.Sequential(
            nn.Linear(combined_obs_act_dim, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, self.nx),
            )

    def forward(
            self,
            x: th.Tensor,
            u: th.Tensor,
    ):
        x = th.cat([x, u], dim=-1)
        return self.net(x)

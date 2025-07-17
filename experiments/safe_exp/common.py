import gymnasium as gym
import numpy as np
import yaml
from x_mpsc import envs

from x_mpsc.algs.pets_torch.utils.wrappers import VideoWrapper

class LinearConstraint:
    """A simple linear constraint"""

    def __init__(self, obs_dim, lower_bound, upper_bound):
        self.obs_dim = obs_dim
        self.a = (upper_bound + lower_bound) / 2
        self.b = (upper_bound - lower_bound) / 2

    def get_value(self, obs):
        return (np.abs(obs[self.obs_dim] - (self.a))) - self.b


class SCGWrapper(gym.Wrapper):
    """A Wrapper that resemble the safe-control-gym interface for constrained enviornments

    - Takes a list of constraints and evaluates them in each step
    - Adds "constraint_values" and "constraint_violation" keys to info
    - by convention, poistive constraint values are violations, negative constraint values mean no violation
    - by default sets done=true if any of the constraints are violated
    """

    def __init__(self, env, constraints=None, done_on_violation=True):
        super().__init__(env=env)
        if constraints is None:
            constraints = []
        self.constraints = constraints
        self.num_constraints = len(constraints)
        self.done_on_violation = done_on_violation

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        c_values = np.array([c.get_value(obs) for c in self.constraints])
        c_violation = any((c_values) > 0)

        if c_violation and self.done_on_violation:
            done = True

        info.update({"constraint_values": c_values, "constraint_violation": int(c_violation)})

        return obs, reward, done, info

    def reset(self):
        obs, _ = self.env.reset()
        info = {}
        info["constraint_values"] = np.array([c.get_value(obs) for c in self.constraints])
        return obs, info


def load_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def make_env(task, task_config, seed=42, record_videos=False):
    # if task == "cartpole":
    #     from safe_control_gym.envs.gym_control.cartpole import CartPole
    #     return CartPole(seed=seed, **task_config)
    # else:
    env = gym.make(task)
    env.seed(seed)

    constraints = []
    for c in task_config.constraints:
        obj = globals().get(c["cls_name"])(**c["kwargs"])
        constraints.append(obj)
    env = SCGWrapper(env, constraints=constraints)
    
    if record_videos:
        env = VideoWrapper(env, episode_trigger=lambda x : True, video_folder=f"./tmp/{task}/")

    return env

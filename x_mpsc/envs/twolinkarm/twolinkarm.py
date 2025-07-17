r"""An environment of a 2-link arm as described in:
    Li and Todorov; Iterative Linear Quadratic Regulator Design for Nonlinear
    Biological Movement Systems

    Author:         Sven Gronauer (sven.gronauer@tum.de)
    Created:        27.07.2020
    Major Changes:  29.09.2020

    TwoLinkArmDynamicsEnv-v0:
        + System parameters: 10cm link length, 1kg link mass
        + No self-collision
        + obs. space: x is dim 10
        + Reward function: || distance_tip_to_goal ||_2^2 - || a_t ||_2^2
"""
import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw


def get_C(a, thetas, theta_dots):
    """A matrix holding centrifugal and Coriolis forces. C has shape 2x1."""
    # sin_theta_2 = np.sin(thetas[1])
    # c_11 = -a[1] * sin_theta_2 * theta_dots[1]
    # c_12 = -a[1] * sin_theta_2 * (theta_dots[0] + theta_dots[0])
    # c_21 = a[1] * sin_theta_2 * theta_dots[0]
    # c_22 = 0

    c_1 = -theta_dots[1] * (2*theta_dots[0] + theta_dots[1])
    c_2 = np.square(theta_dots[0])
    return a[1] * np.sin(thetas[1]) * np.array([c_1, c_2], dtype=np.float32)


def get_M(a, thetas: np.ndarray):
    """The positive definite symmetric inertia matrix. M has shape 2x2."""
    cos_theta_2 = np.cos(thetas[1])
    m_11 = a[0] + 2 * a[1] * cos_theta_2
    m_12 = a[2] + a[1] * cos_theta_2
    m_21 = a[2] + a[1] * cos_theta_2
    m_22 = a[2]
    return np.array([[m_11, m_12], [m_21, m_22]], dtype=np.float32)


class TwoLinkArmDynamics:
    def __init__(
            self,
            dt,
            a,
            lengths,
            constrain=False,
            min_bounds=-1.0,
            max_bounds=1.0,
            **kwargs
    ):
        self.dt = dt
        self.a = a
        self.constrain = constrain
        assert constrain is False, 'Not implemented yet.'
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.lengths = lengths

        self.B = np.array([[0.05, 0.025],
                           [0.025, 0.05]])

    def forward(self, x: np.ndarray, u: np.ndarray):

        new_x = x.copy()
        theta = x[0:2]
        theta_dot = x[2:4]

        if u.ndim == 2:
            u = u.reshape((-1))

        M = get_M(self.a, theta)
        C = get_C(self.a, theta, theta_dot)

        theta_acc = np.linalg.pinv(M) @ (u - C - self.B @ theta_dot)
        new_thetas = theta + self.dt * theta_dot
        new_x[:2] = new_thetas
        new_x[2:4] = np.clip(theta_dot + self.dt * theta_acc, -100, 100)
        # print(f"action;: {u}")
        # print(f"theta_acc: {theta_acc}")
        # print(f"new_x: {new_x}")
        # print("---------")
        return new_x


class TwoLinkArmEnv(gym.Env):
    """An environment of a 2-link arm as described in:
    Li and Todorov; Iterative Linear Quadratic Regulator Design for Nonlinear
    Biological Movement Systems

    State Space: (9-dim)

        Internal self.state is the following:
        x = [theta_1, theta_2, theta_1_dot, theta_2_dot]

    Action Space:
        u = [torque_joint_1, torque_joint_2]


        TwoLinkArmDynamicsEnv-v0:
        + System parameters from Li & Todorov paper
        + No self-collision
        + obs. space: x is dim 10
        + Reward function: || distance_tip_to_goal ||_2^2 - || a_t ||_2^2
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    control_modes = ['torque', 'position']

    def __init__(self, debug=True):

        self.dt = 0.02
        self.debug = debug

        """
        Parameters are adjusted to match approx. TwoLinkArm URDF file
        """
        self.m = m = [1.4, 1.0]  # mass of link in kg
        self.s = s = [0.011,
             0.016]  # distance from the joint center to the center of link mass
        self.l = l = [0.30, 0.30]  # length of link; default in paper: [0.30, 0.33]

        # inertia for rotating links
        self.I = I = [0.025, 0.045]

        self.state = np.zeros((4,))
        self.constrained_controls = True
        self.viewer = None
        self.target_xy = np.zeros(2)  # set by step/reset function
        self.action_dim = 2
        self.state_dim = 4
        self.rendering_scale = 0.2 / np.sum(l)
        self.force_factor = 1

        # self.B = B
        self.lengths = np.array(l)

        self.screen = None
        self.clock = None
        self.screen_dim = 512

        self.ep_len = 0
        self.max_ep_len = 100

        # reduced state consists of [theta_1, theta_2, theta_1_dot, theta_2_dot]
        self.state = np.zeros((self.state_dim,))
        self.goal = np.zeros((2,))

        self.a = np.array(
            [I[0] + I[1] + m[1] * l[0]**2,
             m[1] * l[0] * s[1],
             I[1]])

        self.dynamics = TwoLinkArmDynamics(
            dt=self.dt,
            a=self.a,
            lengths=l
        )
        """ constraints """
        self.pos_limit = 0.50
        self.reset_ball = 0.10

        """ setup costs"""
        self.Q = np.zeros((self.state_dim, self.state_dim))
        self.Q[2, 2] = 1.
        self.Q[3, 3] = 1.
        self.R = 0.01 * np.eye(self.action_dim)

        """ state/action space"""
        # high = 1000 * np.ones(10, dtype=np.float32)
        high = 1e5 * np.ones(8, dtype=np.float32)
        high[0:2] = self.pos_limit
        # high[8:10] = 100
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        # set default action space to be in range [-1, +1]
        action_low = -1.0 * np.ones((self.action_dim,))
        action_high = 1.0 * np.ones((self.action_dim,))
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    @staticmethod
    def cost_func(x, u):
        return np.linalg.norm(x[2:4]) + 1e-3 * np.linalg.norm(u)

    def get_observation(self):
        theta_1, theta_2, theta_1_dot, theta_2_dot = self.state
        end_effector_xy = self.get_end_effector_position()
        # to_target_vec = end_effector_xy - self.goal

        return np.array([
            end_effector_xy[0],
            end_effector_xy[1],
            end_effector_xy[0] - self.goal[0],  # target_x
            end_effector_xy[1] - self.goal[1],  # target_y
            # np.cos(theta_1),
            # np.sin(theta_1),
            # np.cos(theta_2),
            # np.sin(theta_2),
            theta_1 / 10,
            theta_2 / 10,
            theta_1_dot / 10,
            theta_2_dot / 10,
        ], dtype=self.observation_space.dtype)

    def get_end_effector_position(self) -> np.ndarray:
        """
        Calculates the current end-effector position

        Returns
        -------
        position
            (2,) shaped vector of xy position

        """
        theta_1, theta_2 = self.state[:2]
        theta_sum = theta_2 + theta_1
        x_y_1 = np.array([np.cos(theta_1) * self.lengths[0],
                          np.sin(theta_1) * self.lengths[0]])
        x_y_2 = x_y_1 + np.array([np.cos(theta_sum) * self.lengths[1],
                                  np.sin(theta_sum) * self.lengths[1]])
        # print('End-effector pos:', x_y_2)
        # time.sleep(4)
        return x_y_2

    def reset(self):
        self.ep_len = 0
        # force the goal to be in reach of the end-effector
        # set velocities to zero
        self.state = np.zeros(4)
        self.state[:2] = np.random.uniform(low=-3.14, high=3.14, size=2)
        while np.linalg.norm(self.get_end_effector_position()) > self.reset_ball:
            self.state[:2] = np.random.uniform(low=-3.14, high=3.14, size=2)
        self.state[2:4] = np.random.uniform(low=-0.2, high=0.2, size=2)
        self.goal = np.random.uniform(low=-self.pos_limit, high=self.pos_limit, size=2)

        self.target_xy = self.get_end_effector_position()
        obs = self.get_observation()
        return obs

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((155, 155, 155))

        theta_1, theta_2 = self.state[:2]
        theta_sum = theta_1 + theta_2

        cos = np.cos
        sin = np.sin

        # elbow point
        p1 = [self.lengths[0] * cos(theta_1),
              self.lengths[1] * sin(theta_1)]

        # end-effector point:
        p2 = [p1[0] + self.lengths[1] * cos(theta_sum),
              p1[1] + self.lengths[1] * sin(theta_sum)]

        link_pixel_scale = self.screen_dim / 4 / self.lengths[0]
        offset = self.screen_dim // 2

        # colors
        green = (128, 155, 77)
        red = (204, 77, 77)
        black = (0, 0, 0)

        # contraints
        x_up = int(self.pos_limit * link_pixel_scale + offset)
        x = int(offset - self.pos_limit * link_pixel_scale)
        constraints_coords = [(x, x), (x, x_up), (x_up, x_up), (x_up, x)]
        gfxdraw.filled_polygon(self.surf, constraints_coords, (255, 255, 255))

        radius = 10
        rod_width = 16
        l, r, t, b = 0, self.screen_dim / 4, rod_width / 2, -rod_width / 2

        # link 1
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0])
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.filled_polygon(self.surf, transformed_coords, black)

        # link 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(float(np.sum(self.state[:2])))
            c = (c[0] + offset + int(p1[0] * link_pixel_scale), c[1] + offset+ int(p1[1] * link_pixel_scale))
            transformed_coords.append(c)
        gfxdraw.filled_polygon(self.surf, transformed_coords, black)

        # goal
        goal_pos_pixel = (self.goal * link_pixel_scale + offset).astype(np.int32)
        gfxdraw.filled_circle(
            self.surf, goal_pos_pixel[0], goal_pos_pixel[1], radius, green
        )

        points = [(0, 0), p1, p2]
        for (x, y) in points:
            x = int(x * link_pixel_scale) + offset
            y = int(y * link_pixel_scale) + offset
            gfxdraw.filled_circle(
                self.surf, x, y, radius, red
            )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return None

    def inverse_kinmatics(self, xy) -> np.ndarray:
        assert xy.size == 2
        x, y = xy
        gt_theta_2 = self.state[1]
        l_1, l_2 = self.lengths

        if np.linalg.norm(xy) > self.reset_ball:
            pass

        top = np.sum(xy**2)-np.sum(self.lengths**2)
        theta_2 = np.arccos(top / (2*np.prod(self.lengths)))

        if np.abs(theta_2 - gt_theta_2) > np.abs(-theta_2 - gt_theta_2):
            # use negative angle
            theta_2 *= -1
        theta_1 = np.arctan(y/x)-np.arctan((l_2*np.sin(theta_2))/(l_1+l_2*np.cos(theta_2)))
        return np.array([theta_1, theta_2])

    def safe_controller(self, x):
        u = self.action_space.sample()
        next_obs = self.dynamics.forward(x, u)
        next_xy_pos = next_obs[:2]
        if np.linalg.norm(next_xy_pos) > self.reset_ball:
            factor = np.square(self.reset_ball / np.linalg.norm(next_xy_pos))
            desired_xy_pos = factor * next_xy_pos
            theta_ik = self.inverse_kinmatics(desired_xy_pos)
            target_state = np.zeros(4)
            target_state[:2] = theta_ik
            K = np.array([[1, 0.1, 0, 0],
                          [0, 10, 0, 0]])
            u = K @ (target_state - self.state)
        return u

    def step(self, action):
        self.ep_len += 1
        assert (np.isfinite(action).all())
        err_msg = "%r (%s) invalid" % (action, type(action))
        if self.constrained_controls:
            action = np.clip(action, -1., +1.)
            # assert self.action_space.contains(action), err_msg

        u = self.force_factor * action
        reward = -self.cost_func(self.get_observation(), u)

        # track thetas separately to avoid [-pi, pi] range issues
        # since thetas are internally calculated with arctan in dynamics.f()
        # thetas, thetas_dot = (self.state[:2], self.state[2:4])
        self.state = self.dynamics.forward(self.state, u)
        obs = self.get_observation()
        terminated = not self.observation_space.contains(obs)
        truncated = True if self.ep_len >= self.max_ep_len else False
        cost = 1.0 if done else 0.0
        if done:
            reward -= 100
        info = {'cost': cost}

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    # Render CartPoleWithDynamicsEnv
    # env = TwoLinkArmDynamicsEnv()
    env = TwoLinkArmEnv()
    terminated = False
    i = 0
    N = 100
    while not terminated:
        done = False
        x, _ = env.reset()
        i = 0
        while i < N:
            i += 1
            env.render()
            u = env.action_space.sample()
            # u = np.array([-1, 1])
            x, r, terminated, truncated, info = env.step(u)
            # time.sleep(0.01)
            # print(f'i={i} r={r}')

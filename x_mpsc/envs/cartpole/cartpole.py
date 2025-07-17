from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


def deg2rad(x):
    return x * np.pi/180


class CartPoleEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, noise: bool = True, g=10.0):
        self.nx = 4
        self.nu = 1
        self.max_speed = 8.0
        self.max_torque = 1.0
        self.dt = 1/50
        self.g = g
        self.m = 0.33
        self.length = 0.5  # actually half the pole's length
        self.masscart = 1.0
        self.masspole = 0.1
        self.noise = noise

        # Angle at which to fail the episode
        self.theta_threshold_radians = deg2rad(12)
        self.x_threshold = 2.4

        self.ep_len = 0
        self.max_ep_len = 100        

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold,
                1e5,
                self.theta_threshold_radians,
                1e5
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.state = np.zeros(self.nx)

        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        self.isopen = True

        self.seed()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        self.ep_len += 1
        if isinstance(action, np.ndarray):
            action = action.item()

        if self.noise:
            action += 0.005 * np.random.uniform(-self.max_torque, self.max_torque)
        action_clipped = np.clip(action, -1, 1)
        
        x, x_dot, theta, theta_dot = self.state

        gravity = self.g
        length = self.length
        masstotal = (self.masspole + self.masscart)
        force_mag = 10.0
        mpl = self.masspole * self.length
        friction = 0.1
        tau = self.dt

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        u = action_clipped * force_mag

        xdot_update = (-2 * mpl * (
                theta_dot ** 2) * sin_theta + 3 * self.masspole * gravity * sin_theta * cos_theta + 4 * u - 4 * friction * x_dot) / (
                              4 * masstotal - 3 * self.masspole * cos_theta ** 2)
        thetadot_update = (-3 * mpl * (
                theta_dot ** 2) * sin_theta * cos_theta + 6 * masstotal * gravity * sin_theta + 6 * (
                                   u - friction * x_dot) * cos_theta) / (
                                  4 * length * masstotal - 3 * mpl * cos_theta ** 2)

        x_new = x + x_dot * tau
        theta_new = theta + theta_dot * tau
        x_dot_new = x_dot + xdot_update * tau
        theta_dot_new = theta_dot + thetadot_update * tau

        self.state = np.array([x_new, x_dot_new, theta_new, theta_dot_new],
                              dtype=np.float32).flatten()
        obs = self._get_obs()
        terminated = not self.observation_space.contains(obs)
        truncated = True if self.ep_len >= self.max_ep_len else False
        rew = -(np.linalg.norm(obs)**2 + 1e-4 * np.linalg.norm(action)**2)
        rew -= 100 if terminated else 0.0  
        cost = 1.0 if terminated else 0.0
        info = {'cost': cost}
        return self._get_obs(), rew, terminated, truncated, info

    def _get_obs(self):
        scale = 5e-3
        obs = self.state.copy()
        # if self.noise:
        #     obs += np.random.uniform(-scale, scale, size=self.state.size)
        return np.asarray(obs, dtype=self.observation_space.dtype)

    def render(self, mode='human'):
        """Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.

        Returns:
            ndarray: A multidimensional array with the RGB frame captured by PyBullet's camera.

        """
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ValueError(
                "pygame is not installed, run `pip install pygame`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * 1
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def safe_controller(self, x):
        if np.random.uniform(0, 1) > 0.2:
            K = np.array([0.5, 0.5, 100, 30]) * 3.3
            K = np.array([21, -320, 30, -70])
            K = np.array([-3, -18, -113, -40])
            K = np.array([-5, -3, -113, -40])
            u = K @ x #, -1, 1)
        else:
            u = self.action_space.sample()
        return u


    def reset(
        self,
        state: Optional[np.array] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        self.ep_len = 0
        if state is not None and state.shape == (self.nx,):
            self.state = np.asarray(state, dtype=np.float32)
        else:
            high = 0.05 * np.ones(self.nx)
            low = -0.05 * np.ones(self.nx)
            self.state = np.float32(self.np_random.uniform(low=low, high=high))
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == '__main__':
    env = CartPoleEnv(noise=False)
    x, _ = env.reset()
    for _ in range(100):
        done = False
        x, _ = env.reset()
        count = 0
        while not done:
            env.render()
            if hasattr(env, 'safe_controller'):
                a = env.safe_controller(x)
            else:
                a = env.action_space.sample()
            x, r, done, _ = env.step(a)
            count += 1
        print(f"took steps: {count}")

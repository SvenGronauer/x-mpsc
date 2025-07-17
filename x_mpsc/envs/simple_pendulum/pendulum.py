r"""The Simple Pendulum from Underactuated Robotics (Russ Tedrake).

http://underactuated.mit.edu/pend.html
"""

from os import path
from typing import Optional
import pygame
from pygame import gfxdraw
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class SimplePendulumEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, noise: bool = True, g=10.0, render_mode='rgb_array'):
        self.max_speed = 8.0
        self.max_torque = 1.0
        self.dt = 0.1
        self.g = g
        self.m = 0.33
        self.l = 1.0
        self.damping = 0.1
        self.period = 2 * np.pi * np.sqrt(self.l / self.g)  # Period of oscillation
        self.noise = noise
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = np.zeros(2)
        self.last_u = None

        self.obs_dim = 2
        self.act_dim = 1

        self.ep_len = 0
        self.max_ep_len = 100

        self.screen_dim = 512

        state_high = np.array([2.0833*np.pi, self.max_speed], dtype=np.float32)
        state_low = np.array([0.25*np.pi, -self.max_speed], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
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

    def step(self, u):
        self.ep_len += 1
        self.render()
        if isinstance(u, float):
            u = np.array([u])

        # user keyboard inputs
        if pygame.display.get_init() and self.screen is not None:
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[pygame.K_LEFT]:
                u = np.array([-self.max_torque])
            if pressed_keys[pygame.K_RIGHT]:
                u = np.array([self.max_torque])

        if self.noise:
            u += 0.01 * np.random.uniform(-self.max_torque, self.max_torque)
        torque = np.clip(u, -self.max_torque, self.max_torque)
        th, thdot = self.state  # th := theta
        # print(f"thdot: {thdot}")
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        b = self.damping
        self.last_u = torque  # for rendering

        obs = self._get_obs()
        rew = np.cos(obs[0]) - 1e-3 * obs[1]**2 - 1e-3 * np.linalg.norm(u)**2
        acc = (torque + m*g*l*np.sin(th) - b*thdot) / (m*l**2)  # no damping
        newthdot = thdot + acc * dt
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot], dtype=np.float32).flatten()
        terminated = not self.observation_space.contains(self.state)
        truncated = True if self.ep_len >= self.max_ep_len else False
        cost = 1.0 if terminated else 0.0
        info = {'cost': cost}
        return obs, rew, terminated, truncated, info

    def _get_obs(self):
        diff = 5e-3 * (self.observation_space.high - self.observation_space.low)
        obs = self.state.copy()
        if self.noise:
            obs += np.random.uniform(-diff, diff, size=self.state.size)
        return np.asarray(obs, dtype=self.observation_space.dtype)

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_dim, self.screen_dim))
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        # colors
        green = (128, 155, 77)
        red = (204, 77, 77)
        black = (0, 0, 0)

        # draw constraints
        origin = (self.screen_dim // 2, self.screen_dim // 2)
        y1 = self.screen_dim  # // 2 * (1 + np.sin(angle_low))
        y2 = self.screen_dim
        x2 = self.screen_dim // 2 * (1 - np.sin(self.observation_space.high[0] % np.pi))
        constraints_coords = [origin, (0, y1), (0, self.screen_dim), (x2, y2)]
        gfxdraw.filled_polygon(self.surf, constraints_coords, (155, 155, 155))

        fulfills_state_constraints = self.observation_space.contains(self.state)
        color = green if fulfills_state_constraints else red

        rod_length = 1 * scale
        rod_width = 0.1 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, color)
        gfxdraw.filled_polygon(self.surf, transformed_coords, color)

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), color)
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), color
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(1.5*rod_width), color
        )
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(1.5*rod_width), black
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            # print(f"scales: {scale * np.abs(self.last_u) / 2} {scale * np.abs(self.last_u) / 2} ")
            scale_img = pygame.transform.smoothscale(
                img, (scale * float(np.abs(self.last_u)) / 2, scale * float(np.abs(self.last_u)) / 2)
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def reset(
        self,
        *,
        state: Optional[np.array] = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        self.ep_len = 0
        # super().reset(seed=seed)
        if state is not None and state.shape == (2,):
            self.state = np.asarray(state, dtype=np.float32)
        else:
            speed_factor = 0.15
            high = np.array([1.25*np.pi, speed_factor*self.max_speed])
            low = np.array([0.75*np.pi, -speed_factor*self.max_speed])
            self.state = np.float32(self.np_random.uniform(low=low, high=high))
        self.last_u = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


if __name__ == '__main__':
    env = SimplePendulumEnv(noise=False)
    x, _ = env.reset()
    done = False
    while not done:
        x, r, terminated, truncated, _ = env.step(1.0)
        done = terminated or truncated

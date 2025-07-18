import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym

import pybullet as pb
from pybullet_utils import bullet_client
import abc
import xml.etree.ElementTree as etxml

from pybullet_envs.robot_bases import XmlBasedRobot, MJCFBasedRobot, \
    URDFBasedRobot, BodyPart
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
import numpy as np
import pybullet


class WalkerBase(MJCFBasedRobot):

    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        MJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]

        high = 1e5 * np.ones(15, dtype=np.float32)
        # high[3] = 4.0  # x-velocity

        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1),
                                     0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list],
                                     dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(
                self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    def calc_state(self):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array(
            [p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
        )  # torso z is more informative than mean z
        self.body_real_xyz = body_pose.xyz()
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(
            self.walk_target_y - self.body_xyz[1],
            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1],
             self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw

        rot_speed = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw),
                                                                 np.cos(-yaw),
                                                                 0], [0, 0, 1]])
        vx, vy, vz = np.dot(rot_speed,
                            self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array(
            [
                z - self.initial_z,
                np.sin(angle_to_target),
                np.cos(angle_to_target),
                0.3 * vx,
                0.3 * vy,
                0.3 * vz,
                # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                r,
                p
            ],
            dtype=np.float32)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5,
                       +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        debugmode = 0
        if (debugmode):
            print("calc_potential: self.walk_target_dist")
            print(self.walk_target_dist)
            print("self.scene.dt")
            print(self.scene.dt)
            print("self.scene.frame_skip")
            print(self.scene.frame_skip)
            print("self.scene.timestep")
            print(self.scene.timestep)
        return -self.walk_target_dist / self.scene.dt


class Hopper(WalkerBase):
    foot_list = ["foot"]

    def __init__(self):
        WalkerBase.__init__(self, "hopper.xml", "torso", action_dim=3,
                            obs_dim=15, power=0.75)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1


class SafeHopperBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, render=False):
        self.robot = Hopper()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)

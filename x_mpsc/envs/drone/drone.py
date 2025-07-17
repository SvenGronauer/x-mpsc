r"""A 3D Drone Environment for set-point tracking.
"""

import os
import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym

import pybullet as pb
from pybullet_utils import bullet_client
import abc
import xml.etree.ElementTree as etxml

# local imports
from x_mpsc.envs.drone.controller import AttitudeRateController


def get_assets_path() -> str:
    r""" Returns the path to the files located in envs/data."""
    data_path = os.path.join(os.path.dirname(__file__), 'assets')
    return data_path


def deg2rad(x):
    return x * np.pi/180

def rad2deg(x):
    return x * 180 / np.pi


class CrazyFlieAgent(object):
    file_name = "cf21x.urdf"

    def __init__(
            self,
            bc: bullet_client.BulletClient,
            time_step: float,
            noise: bool = True,
            global_scaling=1,
            xyz: np.ndarray = np.array([0., 0., 1.], dtype=np.float64),  # [m]
            rpy: np.ndarray = np.zeros(3, dtype=np.float64),  # [rad]
            xyz_dot: np.ndarray = np.zeros(3, dtype=np.float64),  # [m/s]
            rpy_dot: np.ndarray = np.zeros(3, dtype=np.float64),  # [rad/s]
    ):
        self.bc = bc
        self.body_unique_id = self.load_assets()
        self.force_torque_factor = 1e-2  #todo set me
        self.thrust_factor = 0.075
        self.global_scaling = global_scaling
        self.noise = noise
        self.time_step = time_step

        # state info
        self.xyz = xyz   # [m]
        self.xyz_dot = xyz_dot   # [m/s] world coordinates
        self.rpy = rpy   # [rad]
        self.quaternion = np.array(pb.getQuaternionFromEuler(rpy))
        self.rpy_dot = rpy_dot   # [rad/s] local body frame

        # GUI variables
        self.axis_x = -1
        self.axis_y = -1
        self.axis_z = -1

        # Attitude rate PID controller
        self.pid_controller = AttitudeRateController(bc=bc, time_step=time_step)

    def apply_action(self, action: np.ndarray) -> None:
        """Returns the forces that are applied to drone motors."""

        pwm_values = self.pid_controller.act(action, drone=self)

        motor_forces = self.thrust_factor * (pwm_values / 30000)

        torques = self.force_torque_factor * motor_forces
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        # Set motor forces (thrust) and yaw torque in PyBullet simulation
        self.apply_motor_forces(motor_forces)
        self.apply_z_torque(z_torque)

    def apply_motor_forces(self, forces):
        """Apply a force vector to the drone motors."""
        assert forces.size == 4
        for i in range(4):
            self.bc.applyExternalForce(
                self.body_unique_id,
                i,
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=pb.LINK_FRAME
            )
        # Do motor speed visualization
        for i in range(4):
            self.bc.setJointMotorControl2(bodyUniqueId=self.body_unique_id,
                                    jointIndex=i,
                                    controlMode=pb.VELOCITY_CONTROL,
                                    targetVelocity=10,  # todo
                                    force=0.010)

    def apply_z_torque(self, torque):
        """Apply torque responsible for yaw."""
        self.bc.applyExternalTorque(
            self.body_unique_id,
            4,  # center of mass link
            torqueObj=[0, 0, torque],
            flags=pb.LINK_FRAME
        )

    def get_state(self) -> np.ndarray:
        state = np.concatenate([
            self.xyz,
            self.rpy,
            self.xyz_dot,
            self.rpy_dot,  # local body frame
        ], dtype=np.float32)
        return state.reshape(12, )

    def load_assets(self) -> int:
        """Loads the robot description file into the simulation.

        Expected file format: URDF

        Returns
        -------
            body_unique_id of loaded body
        """
        assert CrazyFlieAgent.file_name.endswith('.urdf')
        fnp = os.path.join(get_assets_path(), CrazyFlieAgent.file_name)
        assert os.path.exists(fnp), f'Did not find {fnp} at: {get_assets_path()}'
        xyz = (0, 0, 0.0125)
        random_rpy = (0, 0, 0)

        body_unique_id = self.bc.loadURDF(
            fnp,
            xyz,
            pb.getQuaternionFromEuler(random_rpy),
            # Important Note: take inertia from URDF...
            # flags=pb.URDF_USE_INERTIA_FROM_FILE
        )
        assert body_unique_id >= 0
        return body_unique_id

    def show_local_frame(self):
        AXIS_LENGTH = 0.1  # todo sven: get right L
        self.axis_x = self.bc.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[AXIS_LENGTH, 0, 0],
            lineColorRGB=[1, 0, 0],
            parentObjectUniqueId=self.body_unique_id,
            parentLinkIndex=-1,
            replaceItemUniqueId=self.axis_x
            )
        self.axis_y = self.bc.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, AXIS_LENGTH, 0],
            lineColorRGB=[0, 1, 0],
            parentObjectUniqueId=self.body_unique_id,
            parentLinkIndex=-1,
            replaceItemUniqueId=self.axis_y,
            )
        self.axis_z = self.bc.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, AXIS_LENGTH],
            lineColorRGB=[0, 0, 1],
            parentObjectUniqueId=self.body_unique_id,
            parentLinkIndex=-1,
            replaceItemUniqueId=self.axis_z
            )

    def update_information(self) -> None:
        r""""Retrieve drone's kinematic information from PyBullet simulation.

            xyz:        [m] in cartesian world coordinates
            rpy:        [rad] in cartesian world coordinates
            xyz_dot:    [m/s] in cartesian world coordinates
            rpy_dot:    [rad/s] in body frame
        """
        bid = self.body_unique_id
        pos, quat = self.bc.getBasePositionAndOrientation(bid)
        self.xyz = np.array(pos, dtype=np.float64)  # [m] in cartesian world coordinates
        self.quaternion = np.array(quat, dtype=np.float64)  # orientation as quaternion
        self.rpy = np.array(self.bc.getEulerFromQuaternion(quat), dtype=np.float64)  # [rad] in cartesian world coordinates

        # PyBullet returns velocities of base in Cartesian world coordinates
        xyz_dot_world, rpy_dot_world = self.bc.getBaseVelocity(bid)
        self.xyz_dot = np.array(xyz_dot_world, dtype=np.float64)  # [m/s] in world frame
        # FIXED: transform omega from world frame to local drone frame
        R = np.asarray(self.bc.getMatrixFromQuaternion(quat)).reshape((3, 3))
        self.rpy_dot = R.T @ np.array(rpy_dot_world, dtype=np.float64)  # [rad/s] in body frame


class DroneEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, use_graphics: bool = False, noise: bool = True):
        self.g = 9.81
        self.m = 0.030
        self.noise = noise

        self.use_graphics = use_graphics
        self.target_pos = np.array([0, 0, 1])
        self.screen_dim = 512

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        bounds = 1e5 * np.ones(12, dtype=np.float32)
        bounds[3] = deg2rad(10)  # roll in [rad]
        bounds[4] = deg2rad(10)  # pitch in [rad]
        self.observation_space = gym.spaces.Box(low=-bounds, high=bounds, dtype=np.float32)
        self.seed()

        # Physics parameters depend on the task
        self.dt = 1./50
        self.number_solver_iterations = 5

        # === Initialize and setup PyBullet ===
        self.bc = self._setup_client_and_physics(use_graphics)
        self.stored_state_id = -1
        self._setup_simulation()
        self.state = self.drone.get_state()

        self.ep_len = 0
        self.max_ep_len = 100

    def _setup_client_and_physics(
            self,
            use_graphics=False
    ) -> bullet_client.BulletClient:
        r"""Creates a PyBullet process instance.

        The parameters for the physics simulation are determined by the
        get_physics_parameters() function.

        Parameters
        ----------
        use_graphics: bool
            If True PyBullet shows graphical user interface with 3D OpenGL
            rendering.

        Returns
        -------
        bc: BulletClient
            The instance of the created PyBullet client process.
        """
        if use_graphics or self.use_graphics:
            bc = bullet_client.BulletClient(connection_mode=pb.GUI)
        else:
            bc = bullet_client.BulletClient(connection_mode=pb.DIRECT)
        bc.setAdditionalSearchPath(get_assets_path())
        # disable GUI debug visuals
        # bc.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, int(use_graphics))
        bc.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSolverIterations=self.number_solver_iterations,
            deterministicOverlappingPairs=1,
            numSubSteps=1)
        bc.setGravity(0, 0, -self.g)
        # bc.setDefaultContactERP(0.9)
        return bc

    def _setup_simulation(self) -> None:
        plane_file_name = "plane2.urdf"
        plane_path = os.path.join(get_assets_path(), plane_file_name)
        assert os.path.exists(plane_path), f'no file found at: {plane_path}'
        self.PLANE_ID = self.bc.loadURDF(plane_file_name)
        room_path = os.path.join(get_assets_path(), "room_10x10.urdf")
        pb.loadURDF(room_path, useFixedBase=True)
        self.drone = CrazyFlieAgent(
            bc=self.bc,
            time_step=self.dt,
        )
        self.bc.changeDynamics(
            bodyUniqueId=self.drone.body_unique_id,
            linkIndex=-1,
            mass=self.m,
        )
        # print(f'Spawn target pos at:', self.target_pos)
        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[0.95, 0.1, 0.05, 0.4],
        )
        # Spawn visual without collision shape
        self.target_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos
        )

    @classmethod
    def safe_controller(cls, x):
        if np.random.uniform(0, 1) < 0.3:
            u = np.random.uniform(-1, 1, 4)
        else:
            thrust = 0.3 * (1 - x[2]) - 0.3 * x[8]
            scale = 0.4
            r = -0.1 * x[3] * 180/np.pi
            p = -0.1 * x[4] * 180/np.pi
            u = np.array([thrust, r, p, 0.]) + np.random.uniform(-scale, scale, 4)
        return u

    def seed(self, seed=None):
        from gymnasium.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray):
        if self.noise:
            scale = 0.01
            action = action + np.random.uniform(-scale, scale, size=action.size)
        self.drone.apply_action(action)
        self.ep_len += 1

        # step simulation once forward and collect information from PyBullet
        self.bc.stepSimulation()
        self.drone.update_information()

        self.state = self.drone.get_state()
        delta = self.target_pos - self.drone.xyz
        terminated = not self.observation_space.contains(self.state)
        truncated = True if self.ep_len >= self.max_ep_len else False
        cost = 1.0 if done else 0.0
        info = {'cost': cost}
        rew = -np.linalg.norm(delta)**2
        rew += -100 if terminated else 1.0

        return self._get_obs(), rew, terminated, truncated, info

    def _get_obs(self):
        diff = 5e-3
        obs = self.state.copy()
        if self.noise:
            obs += np.random.uniform(-diff, diff, size=self.state.size)
        return np.asarray(obs, dtype=self.observation_space.dtype)

    def render(self, mode="human"):
        if mode == 'human':
            if not self.use_graphics:
                self.bc.disconnect()
                self.use_graphics = True
                self.bc = self._setup_client_and_physics(True)
                self._setup_simulation()
                self.drone.show_local_frame()
                # Save the current PyBullet instance as save state
                # => This avoids errors when enabling rendering after training
                self.stored_state_id = self.bc.saveState()

                self.set_camera()
        if mode != "rgb_array":
            return np.array([])
        else:
            raise NotImplementedError

    def reset(self):
        self.ep_len = 0
        # init drone state
        init_xyz = np.array([0, 0, 0.0125], dtype=np.float32)
        # init_xyz = np.array([0, 0, 0.0325], dtype=np.float32)
        init_xyz_dot = np.zeros(3)
        init_rpy_dot = np.zeros(3)

        # sample yaw from [-pi,+pi]
        rpy = np.array([0, 0, np.random.uniform(-np.pi, np.pi)])
        quat = self.bc.getQuaternionFromEuler(rpy)

        xy_lim = 0.25
        pos = init_xyz
        pos[:2] += np.random.uniform(-xy_lim, xy_lim, size=2)

        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos,
            ornObj=quat
        )
        R = np.array(self.bc.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=init_xyz_dot,
            # PyBullet assumes world frame, so local frame -> world frame
            angularVelocity=R.T @ init_rpy_dot
        )
        self.drone.update_information()
        self.state = self.drone.get_state()
        return self._get_obs(), {}

    def set_camera(self):
        if self.use_graphics:
            # === Set camera position
            self.bc.resetDebugVisualizerCamera(
                cameraTargetPosition=(0.0, 0.0, 0.0),
                cameraDistance=1.8,
                cameraYaw=45,
                cameraPitch=-30
            )


if __name__ == '__main__':
    env = DroneEnv(noise=True, use_graphics=True)
    # env.render()
    done = False
    data = []
    for i in range(21):
        steps = 0
        x, _ = env.reset()
        done = False
        while not done:
            # env.render()
            thrust = 0.3 * (1 - x[2]) - 0.2 * x[8] #* np.sign(1 - x[2]) - x[8]
            u = np.array([thrust, 0, 0, 0.]) + np.random.uniform(-0.1, 0.1, 4)
            data.append(x)
            x, r, terminated, truncated, _ = env.step(u)
            done = terminated or truncated
            steps += 1
            time.sleep(1/100)
            if steps >= 200:
                done = True

    # xs = np.array(data)
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    # ax1.plot(rad2deg(xs[:, 3]))
    # ax2.plot(rad2deg(xs[:, 4]))
    # ax3.plot(rad2deg(xs[:, 5]))
    # # ax.scatter(xs[:, 9], xs[:, 11])
    # plt.show()
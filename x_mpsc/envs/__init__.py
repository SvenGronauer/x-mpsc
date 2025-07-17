from gymnasium.envs.registration import register

register(
    id="MassSpringDamper-v0",
    entry_point="x_mpsc.envs.linear.mass_spring_damper:MassSpringDamperEnv",
)

# ==============================================================================
# Pendulum System
register(
    id='SimplePendulum-v0',
    entry_point='x_mpsc.envs.simple_pendulum.pendulum:SimplePendulumEnv',
)
# ==============================================================================
# Reacher Robot
register(
    id='TwoLinkArm-v0',
    entry_point='x_mpsc.envs.twolinkarm.twolinkarm:TwoLinkArmEnv',
)
# ==============================================================================
# CartPole Robot

register(
    id='SafeCartPole-v0',
    entry_point='x_mpsc.envs.cartpole.cartpole:CartPoleEnv',
)


# ==============================================================================
# Drone Robot
register(
    id='SafeDrone-v0',
    entry_point='x_mpsc.envs.drone.drone:DroneEnv',
)

# ==============================================================================
# MuJoCo Environments
register(
    id="SafeHopper-v4",
    entry_point="x_mpsc.envs.mujoco:SafeHopperEnv",
    max_episode_steps=1000,
)


register(
    id="SafeHopper-v0",
    entry_point="x_mpsc.envs.hopper:SafeHopperBulletEnv",
    max_episode_steps=1000,
)

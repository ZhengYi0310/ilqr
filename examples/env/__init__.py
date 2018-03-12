from gym.envs.registration import register

register(
    id='CartPoleContinuous-TensorDynamics',
    entry_point='env.cartpole_continuous:CartPoleContinuousEnv',
    timestep_limit=200,
    reward_threshold=195.0,
)
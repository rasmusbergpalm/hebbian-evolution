from envs.car_racing import CarRacing
import gym

gym.register(
    id='CarRacingCustom-v0',
    entry_point='envs:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)

from typing import Dict

import gym
import torch as t
from evostrat import Individual


class BaseAnt(Individual):
    def __init__(self, env_args: Dict):
        self.env_args = env_args

    def policy(self, x):
        raise NotImplementedError

    def fitness(self, render=False) -> float:
        gym.logger.set_level(40)
        env = gym.make('CustomAntBulletEnv-v0', **self.env_args)
        if render:
            env.render()
        obs = env.reset()

        # Burn-in phase for the quadruped so it starts on the floor
        action = t.zeros(8).numpy()
        for _ in range(40):
            obs, _, _, _ = env.step(action)
            if render:
                env.render()

        r_tot, neg_r, n = (0, 0, 0)
        while True:
            with t.no_grad():
                action = self.policy(t.tensor(obs)).numpy()
            obs, _, done, info = env.step(action)
            r = env.unwrapped.rewards[1]  # Distance walked
            r_tot += r
            if render:
                env.render()
            if n > 200:
                neg_r = neg_r + 1 if r < 0 else 0
                if neg_r > 30 or done:
                    break
            n += 1

        env.close()
        return r_tot

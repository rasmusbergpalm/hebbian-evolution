from typing import Dict

import gym
import numpy as np
import torch as t
from evostrat import Individual

from hebbian_layer import HebbianLayer
# noinspection PyUnresolvedReferences
import pybullet_envs


class HebbianAnt(Individual):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        self.env_args = env_args
        self.params = params

        self.heb1 = HebbianLayer(params["hebb.1"], t.tanh)
        self.heb2 = HebbianLayer(params["hebb.2"], t.tanh)
        self.heb3 = HebbianLayer(params["hebb.3"], t.tanh)

    def net(self, x):
        x = self.heb1.forward(x)
        x = self.heb2.forward(x)
        x = self.heb3.forward(x)
        return x

    def fitness(self, render=False) -> float:
        gym.logger.set_level(40)
        env = gym.make('AntBulletEnv-v0', **self.env_args)
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
            action = self.action(obs)
            if not np.all(np.isfinite(action)):
                print("NaN")
                break
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
        wp = t.cat([self.heb1.h.flatten(), self.heb2.h.flatten(), self.heb3.h.flatten()])
        return r_tot - 0.01 * (wp ** 2).mean()

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        return {
            'hebb.1': (28, 128, 5),
            'hebb.2': (128, 64, 5),
            'hebb.3': (64, 8, 5),
        }

    def action(self, obs):
        with t.no_grad():
            return self.net(t.tensor(obs)).numpy()


if __name__ == '__main__':
    env = gym.make('AntBulletEnv-v0')
    params = {k: t.randn(s) for k, s in HebbianAnt.param_shapes().items()}
    ant = HebbianAnt(params, {})
    print(ant.fitness(True))

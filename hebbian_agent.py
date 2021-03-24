from typing import Dict

import gym
import torch as t
from evostrat import Individual
from gym.wrappers import *

from hebbian_layer import HebbianLayer


def last_act_fn(x):
    return t.tensor((t.tanh(x[0]), t.sigmoid(x[1]), t.sigmoid(x[2])))


class HebbianCarRacingAgent(Individual):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        self.env_args = env_args
        self.params = params
        self.heb1 = HebbianLayer(params["hebb.1"], t.tanh, normalize=True)
        self.heb2 = HebbianLayer(params["hebb.2"], t.tanh, normalize=True)
        self.heb3 = HebbianLayer(params["hebb.3"], last_act_fn, normalize=True)

    def net(self, x):
        x = t.tanh(t.conv2d(x, self.params["cnn.1"]))
        x = t.max_pool2d(x, (2, 2))
        x = t.tanh(t.conv2d(x, self.params["cnn.2"], stride=2))
        x = t.max_pool2d(x, (2, 2))
        x = t.flatten(x, 0)
        x = self.heb1.forward(x)
        x = self.heb2.forward(x)
        x = self.heb3.forward(x)
        return x

    def fitness(self, render=False) -> float:
        gym.logger.set_level(40)
        env = gym.make('CarRacingCustom-v0', **self.env_args)
        env = ResizeObservation(env, 84)
        obs = env.reset()
        done = False
        r_tot = 0
        neg_r = 0
        while not done and neg_r < 20:
            action = self.action(obs)
            obs, r, done, info = env.step(action)
            r_tot += r
            neg_r = neg_r + 1 if r < 0 else 0
            if render:
                env.render()

        env.close()
        return r_tot

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        return {
            'cnn.1': (6, 3, 3, 3),
            'cnn.2': (8, 6, 5, 5),
            'hebb.1': (648, 128, 5),
            'hebb.2': (128, 64, 5),
            'hebb.3': (64, 3, 5),
        }

    def action(self, obs):
        with t.no_grad():
            obs = t.tensor(obs / 255.0, dtype=t.float32).permute((2, 0, 1)).unsqueeze(0)
            return self.net(obs).numpy()

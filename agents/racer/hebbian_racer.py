from typing import Dict

import gym
import torch as t
from gym.wrappers import *

from agents.racer.racer import CarRacingAgent, identity, last_act_fn
from hebbian_layer import HebbianLayer


class HebbianCarRacingAgent(CarRacingAgent):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        super().__init__(env_args)
        self.params = params
        self.heb1 = HebbianLayer(params["hebb.1"], t.tanh, normalize=True)
        self.heb2 = HebbianLayer(params["hebb.2"], t.tanh, normalize=True)
        self.heb3 = HebbianLayer(params["hebb.3"], identity, normalize=True)

    def policy(self, x):
        x = t.tanh(t.conv2d(x, self.params["cnn.1"]))
        x = t.max_pool2d(x, (2, 2))
        x = t.tanh(t.conv2d(x, self.params["cnn.2"], stride=2))
        x = t.max_pool2d(x, (2, 2))
        x = t.flatten(x, 0)
        x = self.heb1.forward(x)
        x = self.heb2.forward(x)
        x = self.heb3.forward(x)
        return last_act_fn(x)

    def fitness(self, render=False) -> float:
        r_tot = super().fitness(render)
        wp = t.cat([self.heb1.h.flatten(), self.heb2.h.flatten(), self.heb3.h.flatten()])
        return r_tot - 0.00 * (wp ** 2).mean()

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        return {
            'cnn.1': (6, 3, 3, 3),
            'cnn.2': (8, 6, 5, 5),
            'hebb.1': (648, 128, 5),
            'hebb.2': (128, 64, 5),
            'hebb.3': (64, 3, 5),
        }


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import envs

    agent = HebbianCarRacingAgent({k: 0.1 * t.randn(s) for k, s in HebbianCarRacingAgent.param_shapes().items()}, {})
    print(agent.fitness(True))

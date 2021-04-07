from typing import Dict

import gym
# noinspection PyUnresolvedReferences
import pybullet_envs
import torch as t

from agents.ant.ant import Ant
from hebbian_layer import HebbianLayer


class HebbianAnt(Ant):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        super().__init__(env_args)
        self.params = params

        self.heb1 = HebbianLayer(params["hebb.1"], t.tanh)
        self.heb2 = HebbianLayer(params["hebb.2"], t.tanh)
        self.heb3 = HebbianLayer(params["hebb.3"], t.tanh)

    def policy(self, x):
        x = self.heb1.forward(x)
        x = self.heb2.forward(x)
        x = self.heb3.forward(x)
        return x

    def fitness(self, render=False) -> float:
        r_tot = super().fitness(render)
        wp = t.cat([self.heb1.h.flatten(), self.heb2.h.flatten(), self.heb3.h.flatten()])
        return r_tot - 0.01 * (wp ** 2).mean()

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        return {
            'hebb.1': (28, 128, 5),
            'hebb.2': (128, 64, 5),
            'hebb.3': (64, 8, 5),
        }


if __name__ == '__main__':
    params = {k: 0.1 * t.randn(s) for k, s in HebbianAnt.param_shapes().items()}
    ant = HebbianAnt(params, {})
    print(ant.fitness(True))

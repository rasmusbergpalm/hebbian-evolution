from typing import Dict

import torch as t
import torch.nn.functional as f

from agents.racer.racer import CarRacingAgent, last_act_fn


class StaticCarRacingAgent(CarRacingAgent):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        super().__init__(env_args)
        self.params = params

    def net(self, x):
        x = t.tanh(t.conv2d(x, self.params["cnn.1"]))
        x = t.max_pool2d(x, (2, 2))
        x = t.tanh(t.conv2d(x, self.params["cnn.2"], stride=2))
        x = t.max_pool2d(x, (2, 2))
        x = t.flatten(x, 0)

        x = t.tanh(f.linear(x, self.params["linear.1"].t()))
        x = t.tanh(f.linear(x, self.params["linear.2"].t()))
        x = last_act_fn(f.linear(x, self.params["linear.3"].t()))

        return x

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        return {
            'cnn.1': (6, 3, 3, 3),
            'cnn.2': (8, 6, 5, 5),
            'linear.1': (648, 128),
            'linear.2': (128, 64),
            'linear.3': (64, 3),
        }


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import envs

    agent = StaticCarRacingAgent({k: 0.1 * t.randn(s) for k, s in StaticCarRacingAgent.param_shapes().items()}, {})
    agent.fitness(True)

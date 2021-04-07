from typing import Dict

import torch as t
import torch.nn.functional as f

from agents.ant.ant import Ant


class StaticAnt(Ant):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        super().__init__(env_args)
        self.params = params

    def policy(self, x):
        x = t.tanh(f.linear(x, self.params["linear.1"].t()))
        x = t.tanh(f.linear(x, self.params["linear.2"].t()))
        x = t.tanh(f.linear(x, self.params["linear.3"].t()))
        return x

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        return {
            'linear.1': (28, 128),
            'linear.2': (128, 64),
            'linear.3': (64, 8),
        }


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import envs

    params = {k: 0.1 * t.randn(s) for k, s in StaticAnt.param_shapes().items()}
    ant = StaticAnt(params, {'morphology_xml': 'ant-long-front.xml'})
    print(ant.fitness(True))

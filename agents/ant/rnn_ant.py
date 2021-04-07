from typing import Dict

import torch as t
import torch.nn.functional as f
from torch import _VF

from agents.ant.base_ant import BaseAnt


class RecurrentAnt(BaseAnt):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        super().__init__(env_args)
        self.params = params
        zeros = t.zeros(1, 128)
        self.hxcx = (zeros, zeros)

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        lstm_hidden_size = 128
        return {
            'lstm.weight_ih': (4 * lstm_hidden_size, 28),
            'lstm.weight_hh': (4 * lstm_hidden_size, lstm_hidden_size),
            'lstm.bias_ih': (4 * lstm_hidden_size,),
            'lstm.bias_hh': (4 * lstm_hidden_size,),
            'linear.2': (lstm_hidden_size, 64),
            'linear.3': (64, 8),
        }

    def policy(self, x):
        (hx, cx) = self.hxcx = _VF.lstm_cell(
            x,
            self.hxcx,
            self.params['lstm.weight_ih'],
            self.params['lstm.weight_hh'],
            self.params['lstm.bias_ih'],
            self.params['lstm.bias_hh']
        )
        x = t.tanh(f.linear(hx.squeeze(), self.params["linear.2"].t()))
        x = t.tanh(f.linear(x, self.params["linear.3"].t()))
        return x


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import envs

    agent = RecurrentAnt({k: 0.1 * t.randn(s) for k, s in RecurrentAnt.param_shapes().items()}, {'morphology_xml': 'ant.xml'})
    agent.fitness(True)

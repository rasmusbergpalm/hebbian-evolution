from typing import Dict

import torch as t
import torch.nn.functional as f
from torch import _VF

from agents.racer.racer import last_act_fn, CarRacingAgent


class RecurrentCarRacingAgent(CarRacingAgent):
    def __init__(self, params: Dict[str, t.Tensor], env_args: Dict):
        super().__init__(env_args)
        self.params = params
        zeros = t.zeros(1, 128)
        self.hxcx = (zeros, zeros)

    @staticmethod
    def param_shapes() -> Dict[str, t.Tensor]:
        lstm_hidden_size = 128
        return {
            'cnn.1': (6, 3, 3, 3),
            'cnn.2': (8, 6, 5, 5),
            'lstm.weight_ih': (4 * lstm_hidden_size, 648),
            'lstm.weight_hh': (4 * lstm_hidden_size, lstm_hidden_size),
            'lstm.bias_ih': (4 * lstm_hidden_size,),
            'lstm.bias_hh': (4 * lstm_hidden_size,),
            'linear.2': (lstm_hidden_size, 64),
            'linear.3': (64, 3),
        }

    def net(self, x):
        x = t.tanh(t.conv2d(x, self.params["cnn.1"]))
        x = t.max_pool2d(x, (2, 2))
        x = t.tanh(t.conv2d(x, self.params["cnn.2"], stride=2))
        x = t.max_pool2d(x, (2, 2))
        x = t.flatten(x, 0)

        (hx, cx) = self.hxcx = _VF.lstm_cell(
            x,
            self.hxcx,
            self.params['lstm.weight_ih'],
            self.params['lstm.weight_hh'],
            self.params['lstm.bias_ih'],
            self.params['lstm.bias_hh']
        )
        x = t.tanh(f.linear(hx.squeeze(), self.params["linear.2"].t()))
        x = last_act_fn(f.linear(x, self.params["linear.3"].t()))

        return x


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import envs

    agent = RecurrentCarRacingAgent({k: 0.1 * t.randn(s) for k, s in RecurrentCarRacingAgent.param_shapes().items()}, {})
    agent.fitness(True)

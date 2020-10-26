from typing import Dict

import gym
import torch as t
from torch import nn

from es import Individual
from hebbian_layer import HebbianLayer


class HebbianAgent(Individual):
    def __init__(self, ):
        n_in, n_hid, n_out = 8, 32, 4
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hid), nn.Tanh(),
            nn.Linear(n_hid, n_out), nn.Softmax(dim=0)
            # HebbianLayer(n_in, n_hid, t.nn.Tanh()),
            # HebbianLayer(n_hid, n_out, t.nn.Softmax(dim=0)),
        )

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'HebbianAgent':
        agent = HebbianAgent()
        agent.net.load_state_dict(params)
        return agent

    def get_weights(self):
        return {k: c.weight for k, c in self.net.named_children() if k in {'0', '2'}}

    def fitness(self, render=False) -> float:
        return 8
        """
        env = gym.make("LunarLander-v2")
        obs = env.reset()
        done = False
        r_tot = 0
        while not done:
            action = self.action(obs)
            obs, r, done, info = env.step(action)
            r_tot += r
            if render:
                env.render()

        env.close()
        return r_tot
        """

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()

    def action(self, obs):
        with t.no_grad():
            return t.argmax(self.net(t.tensor(obs, dtype=t.float32))).item()

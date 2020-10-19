from typing import Dict

import gym
import torch as t
from torch import nn

from es import Individual
from hebbian_layer import HebbianLayer


class HebbianAgent(Individual):
    def __init__(self, ):
        self.env = gym.make("LunarLander-v2")
        n_in, n_hid, n_out = 8, 32, 4
        self.net = nn.Sequential(
            HebbianLayer(n_in, n_hid, t.nn.Tanh()),
            HebbianLayer(n_hid, n_out, t.nn.Softmax(dim=0)),
        )

    @staticmethod
    def from_params(params: Dict[str, t.Tensor]) -> 'HebbianAgent':
        agent = HebbianAgent()
        agent.net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        obs = self.env.reset()
        done = False
        r_tot = 0
        while not done:
            action = self.action(obs)
            obs, r, done, info = self.env.step(action)
            r_tot += r
            if render:
                self.env.render()

        return r_tot

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()

    def action(self, obs):
        with t.no_grad():
            return t.argmax(self.net(t.tensor(obs, dtype=t.float32))).item()

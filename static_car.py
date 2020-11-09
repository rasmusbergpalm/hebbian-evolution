from typing import Dict

import gym
import torch as t
from evostrat import Individual
from gym.wrappers import *
from torch import nn


class StaticCarRacingAgent(Individual):
    def __init__(self, env_args: Dict):
        self.env_args = env_args
        self.net = nn.Sequential(
            # 84, 84, 3
            nn.Conv2d(3, 6, 3, bias=False), nn.Tanh(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 8, 5, 2, bias=False), nn.Tanh(), nn.MaxPool2d(2, 2),
            nn.Flatten(),  # (1, 648)
            nn.Linear(648, 128, bias=False), nn.Tanh(),
            nn.Linear(128, 64, bias=False), nn.Tanh(),
            nn.Linear(64, 3, bias=False)  # (1, 3)
        )

    @staticmethod
    def from_params(params: Dict[str, t.Tensor], env_args: Dict) -> 'StaticCarRacingAgent':
        agent = StaticCarRacingAgent(env_args)
        agent.net.load_state_dict(params)
        return agent

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

    def get_params(self) -> Dict[str, t.Tensor]:
        return self.net.state_dict()

    def action(self, obs):
        with t.no_grad():
            obs = t.tensor(obs / 255.0, dtype=t.float32).permute((2, 0, 1)).unsqueeze(0)
            out = self.net(obs)[0]
            return t.tensor((t.tanh(out[0]), t.sigmoid(out[1]), t.sigmoid(out[2]))).numpy()

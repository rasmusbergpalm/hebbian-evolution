from typing import Dict

import gym
import torch as t
from evostrat import Individual
from gym.wrappers import *
from torch import nn

from hebbian_layer import HebbianLayer


def last_act_fn(x):
    return t.tensor((t.tanh(x[0]), t.sigmoid(x[1]), t.sigmoid(x[2])))


class HebbianCarRacingAgent(Individual):
    def __init__(self, env_args: Dict):
        self.env_args = env_args

        self.net = nn.Sequential(
            # 84, 84, 3
            nn.Conv2d(3, 6, 3, bias=False), nn.Tanh(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 8, 5, 2, bias=False), nn.Tanh(), nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=0),  # (648, )
            HebbianLayer(648, 128, nn.Tanh()),
            HebbianLayer(128, 64, nn.Tanh()),
            HebbianLayer(64, 3, last_act_fn)  # (1, 3)
        )

    @staticmethod
    def from_params(params: Dict[str, t.Tensor], env_args: Dict) -> 'HebbianCarRacingAgent':
        agent = HebbianCarRacingAgent(env_args)
        agent.net.load_state_dict(params)
        return agent

    def fitness(self, render=False) -> float:
        # gym.logger.set_level(40)
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
            return self.net(obs).numpy()


if __name__ == '__main__':
    import envs

    HebbianCarRacingAgent({}).fitness(True)

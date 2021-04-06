from typing import Dict

import gym
import torch as t
from evostrat import Individual
from gym.wrappers import *


def last_act_fn(x):
    return t.tensor((t.tanh(x[0]), t.sigmoid(x[1]), t.sigmoid(x[2])))


def identity(x):
    return x


class CarRacingAgent(Individual):
    def __init__(self, env_args: Dict):
        self.env_args = env_args

    def net(self, x):
        raise NotImplementedError

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

    def action(self, obs):
        with t.no_grad():
            obs = t.tensor(obs / 255.0, dtype=t.float32).permute((2, 0, 1)).unsqueeze(0)
            return self.net(obs).numpy()

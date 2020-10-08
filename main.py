from typing import Dict, Iterable

import gym
import torch as t
import torch.nn as nn

from es import evolve
from hebbian_layer import HebbianLayer


class HebbianAgent:
    def __init__(self, ):
        self.env = gym.make("LunarLander-v2")
        n_in, n_out, n_hid = 8, 4, 16
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hid), nn.Tanh(),
            nn.Linear(n_hid, n_out), nn.Softmax(dim=0),
            # HebbianLayer(n_in, n_hid, t.nn.Tanh(), learn_init=False),
            # HebbianLayer(n_hid, n_out, t.nn.Softmax(dim=0), learn_init=False),
        )

    @staticmethod
    def from_params(params: Iterable[t.Tensor]) -> 'HebbianAgent':
        agent = HebbianAgent()
        state_dict = dict(zip(agent.get_params().keys(), params))
        agent.net.load_state_dict(state_dict)
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


"""
if __name__ == '__main__':
    good = t.load('good.t')
    for i in range(100):
        HebbianAgent.from_params(good).fitness(render=True)
"""

if __name__ == '__main__':
    def fitness_fn(params: Iterable[t.Tensor]):
        return HebbianAgent.from_params(params).fitness()


    def eval_best(best: Iterable[t.Tensor], fit):
        if fit > 150:
            t.save(best, "good-mlp.t")


    initial = HebbianAgent().get_params().values()
    best, best_fit = evolve(fitness_fn, initial, 100, 200, 1.0, 0.2, eval_best)
    print(best_fit)

from multiprocessing.pool import Pool
from typing import Dict, Iterable

import gym
import torch as t
import torch.nn as nn
from torch.distributions import Normal
import tqdm
from torch.optim import SGD, Adam

from es import evolve, PopulationDistribution, Individual, es_grads
from hebbian_layer import HebbianLayer


class HebbianAgent(Individual):
    def __init__(self, ):
        self.env = gym.make("LunarLander-v2")
        n_in, n_out = 8, 4
        self.net = nn.Sequential(
            HebbianLayer(n_in, n_out, t.nn.Softmax(dim=0), learn_init=False),
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


"""
if __name__ == '__main__':
    good = t.load('good.t')
    for i in range(100):
        HebbianAgent.from_params(good).fitness(render=True)
"""


class HebbianPopulationDistribution(PopulationDistribution):
    def __init__(self, parameter_distributions: Dict[str, t.distributions.Normal]):
        self.parameter_distributions = parameter_distributions

    def parameters(self) -> Iterable[t.Tensor]:
        return [p.mean for p in self.parameter_distributions.values()]

    def sample(self, n) -> Iterable[HebbianAgent]:
        return [
            HebbianAgent.from_params({
                k: dist.sample() for k, dist in self.parameter_distributions.items()
            })
            for _ in range(n)
        ]

    def log_prob(self, individual: HebbianAgent) -> float:
        return sum([self.parameter_distributions[k].log_prob(p).sum() for k, p in individual.get_params().items()])


if __name__ == '__main__':
    sigma = 1.0
    pop_dist = HebbianPopulationDistribution({k: Normal(t.zeros(v.shape, requires_grad=True), sigma) for k, v in HebbianAgent().get_params().items()})

    iterations = 100
    pop_size = 200
    pool = Pool(8)

    optim = Adam(pop_dist.parameters(), lr=0.1)
    pbar = tqdm.tqdm(range(iterations))
    for _ in pbar:
        optim.zero_grad()
        avg_fitness = es_grads(pop_dist, pop_size, pool)
        optim.step()
        pbar.set_description("avg fit: %.3f" % avg_fitness)
        # ind = pop_dist.sample(1)[0]
        # ind.fitness(render=True)

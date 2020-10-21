from typing import Iterable

import torch as t
import torch.distributions as d

from es import Population
from hebbian_agent import HebbianAgent


class NormalHebbianPopulation(Population):
    def __init__(self, scale):
        self.scale = scale
        self.learning_rule_means = {k: t.randn(v.shape, requires_grad=True) for k, v in HebbianAgent().get_params().items()}

    def parameters(self) -> Iterable[t.Tensor]:
        return list(self.learning_rule_means.values())

    def sample(self, n) -> Iterable[HebbianAgent]:
        agents = []
        for _ in range(n // 2):
            noise = {k: d.Normal(loc=t.zeros_like(v), scale=self.scale).sample() for k, v in self.learning_rule_means.items()}
            agents.append(HebbianAgent.from_params({k: self.learning_rule_means[k] + n for k, n in noise.items()}))
            agents.append(HebbianAgent.from_params({k: self.learning_rule_means[k] - n for k, n in noise.items()}))

        return agents

    def log_prob(self, individual: HebbianAgent) -> t.Tensor:
        return sum([d.Normal(self.learning_rule_means[k], scale=self.scale).log_prob(p).sum() for k, p in individual.get_params().items()])

    def load(self, fname):
        self.learning_rule_means = t.load(fname)

    def save(self, fname):
        t.save(self.learning_rule_means, fname)

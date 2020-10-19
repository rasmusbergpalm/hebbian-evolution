from typing import Iterable

import torch as t
import torch.distributions as d

from es import Population
from hebbian_agent import HebbianAgent


class NormalHebbianPopulation(Population):
    def __init__(self, scale):
        self.p_h = {k: d.Normal(loc=t.randn(v.shape, requires_grad=True), scale=scale) for k, v in HebbianAgent().get_params().items()}

    def parameters(self) -> Iterable[t.Tensor]:
        return [p.mean for p in self.p_h.values()]

    def sample(self, n) -> Iterable[HebbianAgent]:
        return [
            HebbianAgent.from_params({
                k: dist.sample() for k, dist in self.p_h.items()
            })
            for _ in range(n)
        ]

    def log_prob(self, individual: HebbianAgent) -> float:
        return sum([self.p_h[k].log_prob(p).sum() for k, p in individual.get_params().items()])

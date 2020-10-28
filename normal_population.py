from typing import Iterable, Dict, Callable

import torch as t
import torch.distributions as d

from es import Population, Individual


class NormalPopulation(Population):
    def __init__(self, params: Dict[str, t.Tensor], constructor: Callable[[Dict[str, t.Tensor]], Individual], scale: float):
        assert scale > 0, "scale must be greater than zero"
        self.scale = scale
        self.param_means = {k: t.randn(v.shape, requires_grad=True) for k, v in params.items()}
        self.constructor = constructor

    def parameters(self) -> Iterable[t.Tensor]:
        return list(self.param_means.values())

    def sample(self, n) -> Iterable[Individual]:
        samples = []
        for _ in range(n // 2):
            noise = {k: d.Normal(loc=t.zeros_like(v), scale=self.scale).sample() for k, v in self.param_means.items()}

            samples.append((
                self.constructor({k: self.param_means[k] + n for k, n in noise.items()}),
                sum([d.Normal(self.param_means[k], scale=self.scale).log_prob(self.param_means[k] + n).sum() for k, n in noise.items()])
            ))
            samples.append((
                self.constructor({k: self.param_means[k] - n for k, n in noise.items()}),
                sum([d.Normal(self.param_means[k], scale=self.scale).log_prob(self.param_means[k] - n).sum() for k, n in noise.items()])
            ))

        return samples

    def load(self, fname):
        self.param_means = t.load(fname)

    def save(self, fname):
        t.save(self.param_means, fname)

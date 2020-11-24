from typing import Iterable, Tuple, Dict, Callable, Union

import torch as t
import torch.distributions as d
from evostrat import Population, Individual, NormalPopulation


class RandomSharedPopulation(Population):
    def __init__(self,
                 normal_shapes: Dict[str, t.Size],
                 shared_shapes: Dict[str, t.Size],
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual],
                 std: Union[float, str],
                 shared_normal_shape: t.Size,
                 device="cpu"
                 ):
        self.individual_constructor = individual_constructor
        self.normal_pop = NormalPopulation(normal_shapes, lambda x: x, std, False, device=device)

        self.assignments = {k: d.Categorical(logits=t.zeros(shape + (shared_normal_shape[0],))).sample() for k, shape in shared_shapes.items()}
        self.component_means = t.randn(shared_normal_shape, requires_grad=True, device=device)

        self.std = std

    def parameters(self) -> Iterable[t.Tensor]:
        return list(self.normal_pop.parameters()) + [self.component_means]

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        norms = self.normal_pop.sample(n)
        gmms = self.sample_assigned(n)

        return ((self.individual_constructor({**norm_param, **gmm_param}), norm_logp + gmm_logp) for (norm_param, norm_logp), (gmm_param, gmm_logp) in zip(norms, gmms))

    def sample_assigned(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        for i in range(n):
            log_p = 0.0
            params = {}
            for k, idx in self.assignments.items():
                dist = d.Normal(loc=self.component_means[idx], scale=self.std)
                with t.no_grad():
                    sample = dist.sample()
                params[k] = sample
                log_p += dist.log_prob(sample).sum()

            yield params, log_p


if __name__ == '__main__':
    n_rules = 713


    class Fit:
        def fitness(self):
            return 1.0


    pop = RandomSharedPopulation(normal_shapes={'a': (11, 13)},
                                 shared_shapes={'1.h': (648, 128), '2.h': (128, 64), '3.h': (64, 3)},
                                 individual_constructor=lambda x: Fit(),
                                 std=0.1,
                                 shared_normal_shape=(n_rules, 5)
                                 )
    pop.fitness_grads(200)
    i = 0

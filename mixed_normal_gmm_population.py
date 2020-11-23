from typing import Iterable, Tuple, Dict, Callable, Union

import torch as t
from evostrat import Population, Individual, NormalPopulation, GaussianMixturePopulation


class MixedNormalAndGMMPopulation(Population):
    def __init__(self,
                 normal_shapes: Dict[str, t.Size],
                 gmm_shapes: Dict[str, t.Size],
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual],
                 std: Union[float, str],
                 n_components: t.Size,
                 device="cpu"
                 ):
        self.individual_constructor = individual_constructor
        self.normal_pop = NormalPopulation(normal_shapes, lambda x: x, std, False, device=device)
        self.gmm_pop = GaussianMixturePopulation(gmm_shapes, n_components, lambda x: x, std, device=device)

    def parameters(self) -> Iterable[t.Tensor]:
        return list(self.normal_pop.parameters()) + list(self.gmm_pop.parameters())

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        norms = self.normal_pop.sample(n)
        gmms = self.gmm_pop.sample(n)

        return ((self.individual_constructor({**norm_param, **gmm_param}), norm_logp + gmm_logp) for (norm_param, norm_logp), (gmm_param, gmm_logp) in zip(norms, gmms))


if __name__ == '__main__':
    n_rules = 16
    pop = MixedNormalAndGMMPopulation(normal_shapes={},
                                      gmm_shapes={'1.h': (648, 128), '2.h': (128, 64), '3.h': (64, 3)},
                                      individual_constructor=lambda x: x,
                                      std=0.1,
                                      n_components=(n_rules, 5)
                                      )
    pop.fitness_grads(200)
    i = 0

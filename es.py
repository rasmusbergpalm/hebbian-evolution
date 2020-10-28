from typing import Iterable, Callable, Tuple

import torch as t
from torch.multiprocessing import Pool


class Individual:
    """
    An individual which has a fitness. See Population
    """

    def fitness(self) -> float:
        """
        :return: The fitness of the individual. Does not need to be differentiable.
        """
        raise NotImplementedError


class Population:
    """
    A parameterized distribution over individuals. See es_grads.
    """

    def parameters(self) -> Iterable[t.Tensor]:
        """
        :return: The parameters of this population distribution.
        """

        raise NotImplementedError

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        """
        Sample n individuals and compute their log probabilities. The log probability computation must be differentiable.

        :param n: How many individuals to sample
        :return: n individuals and their log probability of being sampled: [(ind_1, log_prob_1), ..., (ind_n, log_prob_n)]
        """
        raise NotImplementedError

    def save(self, fname):
        raise NotImplementedError

    def load(self, fname):
        raise NotImplementedError


def _fitness_fn_no_grad(ind: Individual):
    with t.no_grad():
        return ind.fitness()


def es_grads(
        population: Population,
        n_samples: int,
        pool: Pool = None,
        fitness_shaping_fn: Callable[[Iterable[float]], Iterable[float]] = lambda x: x
):
    """
    Computes the (approximate) negative gradients of the expected fitness of the population
    w.r.t the population parameters. Can be used with the `torch.optim` optimizers.

    pop = PopulationImpl(...)
    optim = torch.optim.Adam(pop.parameters())
    for i in range(N):
        optim.zero_grads()
        es_grads(pop, 200)
        optim.step()

    Uses torch autodiff to compute the gradients. The Individual.fitness does not need to be differentiable,
    but the log probability computations in Population.sample must be.

    Math:

    In evolutionary strategies the goal is to maximize the expected fitness of a distribution of individuals,
    $\max_\theta \mathbb{E}_{z \sim p(z|\theta)} F(z)$, where $F(z)$ is the fitness of an individual $z$ and
    $\theta$ parameterize this distribution. The gradient of this objective is computed using the score function estimator

    \nabla_\theta \mathbb{E}_{z\sim p(z|\theta)} F(z) &= \mathbb{E}_{z\sim p(z|\theta)} F(z) \nabla_\theta \log(p(z|\theta))

    :param population: The population distribution that individuals are sampled from.
    :param n_samples: How many individuals to sample to approximate the gradient
    :param pool: Optional process pool to use when computing the fitness of the sampled individuals.
    :param fitness_shaping_fn: Optional function to modify the fitness, e.g. normalization, etc. Input is a list of n raw fitness floats. Output must also be n floats.
    :return: A (n,) tensor containing the raw fitness (before fitness_shaping_fn) for the n individuals.
    """
    individuals, log_probs = zip(*population.sample(n_samples))
    assert all(lp.ndim == 0 and lp.isfinite() for lp in log_probs), "log_probs must be finite scalars"

    if pool is not None:
        raw_fitness = pool.map(_fitness_fn_no_grad, individuals)
    else:
        raw_fitness = list(map(_fitness_fn_no_grad, individuals))

    fitness = fitness_shaping_fn(raw_fitness)
    t.mean(t.stack([(-ind_fitness * log_prob) for log_prob, ind_fitness in zip(log_probs, fitness)])).backward()
    return t.tensor(raw_fitness)

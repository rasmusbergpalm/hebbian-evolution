from typing import Iterable, Callable, Tuple

import torch as t
from torch.multiprocessing import Pool


class Individual:
    """

    """

    def fitness(self) -> float:
        """
        :return: the fitness of the individual.
        """
        raise NotImplementedError


class Population:
    """

    """

    def parameters(self) -> Iterable[t.Tensor]:
        """

        :return: The parameters of this population distribution. Optimize these using the torch.optim optimizers
        """

        raise NotImplementedError

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        """
        Sample n individuals.

        :param n: How many individuals to sample
        :return: n individuals and their log probability of being sampled
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
    Compute the (approximate) negative gradient of the expected fitness of the population.

    Uses torch autodiff to populate the gradients on all tensors that are used in the
    computation of the log probability of the individuals.

    :param population: The population distribution that individuals are sampled from.
    :param n_samples: How many samples to use to approximate the gradient
    :param pool: A worker pool to use when computing the fitness of the sampled individuals.
    :param fitness_shaping_fn:
    :return:
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

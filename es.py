from typing import Iterable, Callable

import torch as t
from torch.multiprocessing import Pool


class Individual:
    def fitness(self) -> float:
        raise NotImplementedError


class Population:
    def parameters(self) -> Iterable[t.Tensor]:
        raise NotImplementedError

    def sample(self, n) -> Iterable[Individual]:
        raise NotImplementedError

    def log_prob(self, individual: Individual) -> float:
        raise NotImplementedError

    def save(self, fname):
        raise NotImplementedError

    def load(self, fname):
        raise NotImplementedError


def _fitness_fn_no_grad(ind: Individual):
    with t.no_grad():
        return ind.fitness()


def es_grads(
        pop_dist: Population,
        pop_size: int,
        pool: Pool,
        fitness_shaping_fn: Callable[[Iterable[float]], Iterable[float]] = lambda x: x
):
    population = pop_dist.sample(pop_size)
    raw_fitness = pool.map(_fitness_fn_no_grad, population)
    pop_fitness = fitness_shaping_fn(raw_fitness)
    t.mean(t.stack([(-ind_fitness * pop_dist.log_prob(ind)) for ind, ind_fitness in zip(population, pop_fitness)])).backward()
    return sum(raw_fitness) / pop_size

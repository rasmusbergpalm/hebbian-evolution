from typing import Callable, Iterable

import torch as t
import tqdm
from torch.multiprocessing import Pool


class Individual:
    def fitness(self) -> float:
        raise NotImplementedError


class PopulationDistribution:
    def sample(self, n) -> Iterable[Individual]:
        raise NotImplementedError

    def log_prob(self, individual: Individual) -> float:
        raise NotImplementedError


def fitness_fn_no_grad(ind: Individual):
    with t.no_grad():
        return ind.fitness()


def es_grads(
        pop_dist: PopulationDistribution,
        pop_size: int,
        pool: Pool
):
    population = pop_dist.sample(pop_size)
    pop_fitness = pool.map(fitness_fn_no_grad, population)
    t.mean(t.stack([(-ind_fitness * pop_dist.log_prob(ind)) for ind, ind_fitness in zip(population, pop_fitness)])).backward()
    return sum(pop_fitness)/pop_size


def evolve(
        fitness_fn: Callable[[Iterable[t.Tensor]], float],
        initial: Iterable[t.Tensor],
        iterations: int,
        pop_size: int,
        noise: float,
        learning_rate: float,
        callback: Callable[[Iterable[t.Tensor]], None]
):
    eps = 1e-6
    n_eval = 10
    best = initial
    best_fit = fitness_fn(initial)
    pbar = tqdm.tqdm(range(iterations))

    def eval(best):
        return sum([fitness_fn(best) for _ in range(n_eval)]) / n_eval

    for _ in pbar:
        population = [tuple(t.randn(p.shape) for p in best) for _ in range(pop_size // 2)]  # (pop_size, (*param_size))
        population = population + [tuple(-p for p in ind) for ind in population]  # Mirrored sampling

        try_pop = [tuple(b + noise * p for b, p in zip(best, ind)) for ind in population]
        with Pool(8) as p:
            fitness = t.tensor(p.map(fitness_fn, try_pop))
        # fitness = t.tensor([fitness_fn(ind) for ind in try_pop])  # (pop_size, )
        norm_fitness = (fitness - fitness.mean()) / (eps + fitness.std())  # (pop_size, )
        fitness_weighted_sum_pop = tuple(sum([fit * ind[i] for ind, fit in zip(population, norm_fitness)]) for i in range(len(population[0])))
        best = tuple(bp + learning_rate / (pop_size * noise) * fwp for bp, fwp in zip(best, fitness_weighted_sum_pop))
        best_fit = eval(best)

        pbar.set_description("best %0.3f" % best_fit)
        callback(best, best_fit)

    return best, best_fit
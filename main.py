from multiprocessing.pool import Pool

import torch as t
import tqdm
from torch.optim import SGD

from es import es_grads
from gmm_hebbian_population import GMMHebbianPopulation
from normal_hebbian_population import NormalHebbianPopulation
from util import get_writers

if __name__ == '__main__':
    train_writer, test_writer = get_writers('hebbian')

    scale = 1.0
    population = NormalHebbianPopulation(scale)

    learning_rate = 0.2
    iterations = 300
    pop_size = 200
    pool = Pool(8)

    optim = SGD(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        avg_fitness = es_grads(population, pop_size, pool)
        train_writer.add_scalar('fitness', avg_fitness, i)
        optim.step()
        pbar.set_description("avg fit: %.3f" % avg_fitness)
        t.save(optim.state_dict(), 'latest.t')

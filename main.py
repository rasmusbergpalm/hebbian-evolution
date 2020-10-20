from multiprocessing.pool import Pool

import tqdm
from torch.optim import SGD

import util
from es import es_grads
from normal_hebbian_population import NormalHebbianPopulation

if __name__ == '__main__':
    train_writer, test_writer = util.get_writers('hebbian')

    scale = 0.1
    population = NormalHebbianPopulation(scale)

    learning_rate = 0.2
    iterations = 300
    pop_size = 200


    optim = SGD(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        with Pool(8) as pool:
            avg_fitness = es_grads(population, pop_size, pool, util.compute_centered_ranks)
        train_writer.add_scalar('fitness', avg_fitness, i)
        optim.step()
        pbar.set_description("avg fit: %.3f" % avg_fitness)
        population.save('latest.t')

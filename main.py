from multiprocessing.pool import Pool

import tqdm
from torch.optim import Adam

import util
from es import es_grads
from gmm_hebbian_population import GMMHebbianPopulation
from normal_hebbian_population import NormalHebbianPopulation

if __name__ == '__main__':
    train_writer, test_writer = util.get_writers('hebbian')

    scale = 0.1
    num_learning_rules = 2
    population = NormalHebbianPopulation(scale)

    learning_rate = 0.1
    iterations = 30_000
    pop_size = 200

    optim = Adam(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        with Pool(1) as pool:
            rft = es_grads(population, pop_size, pool, util.compute_centered_ranks)

        avg_fit = rft.mean()
        train_writer.add_scalar('fitness', avg_fit, i)
        train_writer.add_scalar('fitness/std', rft.std(), i)
        # for key, ent in population.average_mixing_entroy().items():
        #   train_writer.add_scalar('entropy/%s' % key, ent, i)
        optim.step()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (avg_fit.item(), rft.std().item()))
        population.save('latest.t')
        if avg_fit > 200:
            print("Solved.")
            break

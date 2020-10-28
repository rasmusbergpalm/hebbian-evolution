import tqdm
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam

import util
from es import es_grads
from normal_population import NormalPopulation
from static_car import StaticCarRacingAgent

if __name__ == '__main__':
    set_start_method('spawn')
    train_writer, test_writer = util.get_writers('hebbian')

    scale = 0.1
    agent = StaticCarRacingAgent()
    population = NormalPopulation(
        agent.get_params(),
        StaticCarRacingAgent.from_params,
        scale
    )

    learning_rate = 0.1
    iterations = 30_000
    pop_size = 3

    optim = Adam(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        with Pool(2) as pool:
            fitness = es_grads(population, pop_size, pool, util.compute_centered_ranks)

        avg_fit = fitness.mean()
        train_writer.add_scalar('fitness', avg_fit, i)
        train_writer.add_scalar('fitness/std', fitness.std(), i)
        # for key, ent in population.average_mixing_entroy().items():
        #   train_writer.add_scalar('entropy/%s' % key, ent, i)
        optim.step()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (avg_fit.item(), fitness.std().item()))
        population.save('latest.t')
        if avg_fit > 200:
            print("Solved.")
            break

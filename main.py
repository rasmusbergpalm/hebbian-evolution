import torch as t
import tqdm
from evostrat import NormalPopulation, compute_centered_ranks
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam

import util
from meta_agent import MetaAgent
from rnn_car import RecurrentCarRacingAgent

if __name__ == '__main__':
    set_start_method('spawn')
    train_writer, test_writer = util.get_writers('hebbian')

    env_args = [
        {},
        {'side_force': 1.0},
        {'side_force': -1.0},
        {'friction': 0.5},
        # {'friction': 2.0}
    ]


    def constructor(params) -> MetaAgent:
        return MetaAgent([RecurrentCarRacingAgent.from_params(params, env_arg) for env_arg in env_args])



    agent = RecurrentCarRacingAgent(env_args[0])
    shapes = {k: p.shape for k, p in agent.get_params().items()}
    population = NormalPopulation(shapes, constructor, std=0.1)

    iterations = 30_000
    pop_size = 200

    optim = Adam(population.parameters(), lr=0.1)
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optim.zero_grad()
        with Pool() as pool:
            raw_fitness = population.fitness_grads(pop_size, pool, compute_centered_ranks)

        train_writer.add_scalar('fitness', raw_fitness.mean(), i)
        train_writer.add_scalar('fitness/std', raw_fitness.std(), i)
        optim.step()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (raw_fitness.mean().item(), raw_fitness.std().item()))
        if raw_fitness.mean() > 700:
            t.save(population.parameters(), 'sol.t')
            print("Solved.")
            break

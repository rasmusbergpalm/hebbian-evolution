import torch as t
import tqdm
from evostrat import NormalPopulation, compute_centered_ranks
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam

# noinspection PyUnresolvedReferences
import envs
import util
from meta_agent import MetaAgent
from static_car import StaticCarRacingAgent

if __name__ == '__main__':
    set_start_method('spawn')
    t.multiprocessing.set_sharing_strategy('file_system')

    train_writer, test_writer = util.get_writers('hebbian')

    env_args = [
        {},
        # {'side_force': 10.0},
        {'side_force': -10.0},
        {'friction': 0.5},
        {'friction': 2.0}
    ]


    def constructor(params) -> MetaAgent:
        return MetaAgent([StaticCarRacingAgent.from_params(params, env_arg) for env_arg in env_args])


    agent = StaticCarRacingAgent(env_args[0])
    shapes = {k: p.shape for k, p in agent.get_params().items()}
    population = NormalPopulation(shapes, constructor, std=0.1)

    iterations = 1_000
    pop_size = 200

    optim = Adam(population.parameters(), lr=0.1)
    pbar = tqdm.tqdm(range(iterations))
    best_so_far = -1e9
    for i in pbar:
        optim.zero_grad()
        with Pool() as pool:
            raw_fitness = population.fitness_grads(pop_size, pool, compute_centered_ranks)

        train_writer.add_scalar('fitness', raw_fitness.mean(), i)
        train_writer.add_scalar('fitness/std', raw_fitness.std(), i)
        optim.step()
        mean_fit = raw_fitness.mean().item()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (mean_fit, raw_fitness.std().item()))

        if mean_fit > best_so_far:
            best_so_far = mean_fit
            t.save(population.parameters(), 'best.t')
            util.upload_results('best.t')

        if mean_fit > 900:
            t.save(population.parameters(), 'sol.t')
            util.upload_results('sol.t')
            print("Solved.")
            break

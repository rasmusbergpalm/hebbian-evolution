import torch as t
import tqdm
from evostrat import NormalPopulation, compute_centered_ranks
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam
import util
from hebbian_agent import HebbianCarRacingAgent
from meta_agent import MetaAgent
from mixed_normal_gmm_population import MixedNormalAndGMMPopulation
from static_car import StaticCarRacingAgent

# noinspection PyUnresolvedReferences
import envs

if __name__ == '__main__':
    set_start_method('fork')
    t.multiprocessing.set_sharing_strategy('file_system')

    train_writer, test_writer = util.get_writers('hebbian')
    device = "cuda" if t.cuda.is_available() else "cpu"

    agent = HebbianCarRacingAgent
    env_args = [
        # {},
        {'side_force': 10.0},
        {'side_force': -10.0},
        {'friction': 0.5},
        {'friction': 2.0}
    ]


    def constructor(params) -> MetaAgent:
        return MetaAgent([agent.from_params(params, env_arg) for env_arg in env_args])


    rho=128
    shapes = {k: p.shape for k, p in agent({}).get_params().items()}
    norm_shapes = {k: v for k, v in shapes.items() if not k.endswith('.h')}
    gmm_shapes = {k: v[:-1] for k, v in shapes.items() if k.endswith('.h')}
    n_rules = int(sum([s.numel() for s in gmm_shapes.values()]) / rho)
    population = MixedNormalAndGMMPopulation(norm_shapes, gmm_shapes, constructor, 0.1, (n_rules, 5), device)

    iterations = 1_000
    pop_size = 100

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

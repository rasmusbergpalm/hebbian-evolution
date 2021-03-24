from typing import Dict

import torch as t
import tqdm
from evostrat import NormalPopulation, compute_centered_ranks
from torch.multiprocessing import Pool, set_start_method, cpu_count
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiplicativeLR
import util
from hebbian_agent import HebbianCarRacingAgent
from meta_agent import MetaAgent
from mixed_normal_gmm_population import MixedNormalAndGMMPopulation
from random_shared_population import RandomSharedPopulation
from static_car import StaticCarRacingAgent

# noinspection PyUnresolvedReferences
import envs

if __name__ == '__main__':
    set_start_method('fork')
    t.multiprocessing.set_sharing_strategy('file_system')

    device = "cuda" if t.cuda.is_available() else "cpu"

    agent = HebbianCarRacingAgent
    env_args = [
        {},
        # {'side_force': 10.0},
        # {'side_force': -10.0},
        # {'friction': 0.5},
        # {'friction': 2.0}
    ]

    all_params = agent({}).get_params()
    cnn_params = {k: t.randn(p.shape) for k, p in all_params.items() if not k.endswith('.h')}
    hebb_shapes = {k: p.shape for k, p in all_params.items() if k.endswith('.h')}


    def constructor(hebb_params: Dict) -> MetaAgent:
        params = dict(**cnn_params, **hebb_params)
        return MetaAgent([agent.from_params(params, env_arg) for env_arg in env_args])


    # rho = 0.5
    # norm_shapes = {k: v for k, v in shapes.items() if not k.endswith('.h')}
    # gmm_shapes = {k: v[:-1] for k, v in shapes.items() if k.endswith('.h')}
    # n_rules = int(sum([s.numel() for s in gmm_shapes.values()]) / rho)
    # population = RandomSharedPopulation(norm_shapes, gmm_shapes, constructor, 0.1, (n_rules, 5), device)

    population = NormalPopulation(hebb_shapes, constructor, 0.1, True)
    population.param_means = {k: t.randn(shape, requires_grad=True, device=device) for k, shape in hebb_shapes.items()}  # pop mean init hack

    iterations = 300
    pop_size = 200

    optim = SGD(population.parameters(), lr=0.2)
    sched = MultiplicativeLR(optim, lr_lambda=lambda step: 0.995)
    pbar = tqdm.tqdm(range(iterations))
    best_so_far = -1e9
    train_writer, test_writer = util.get_writers('hebbian')
    for i in pbar:
        optim.zero_grad()
        with Pool(cpu_count() // 2) as pool:
            raw_fitness = population.fitness_grads(pop_size, pool, compute_centered_ranks)

        train_writer.add_scalar('fitness', raw_fitness.mean(), i)
        train_writer.add_scalar('fitness/std', raw_fitness.std(), i)
        optim.step()
        sched.step()
        population.param_logstds = {k: t.log(t.exp(logstd) * 0.999) for k, logstd in population.param_logstds.items()}  # sigma decay hack
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

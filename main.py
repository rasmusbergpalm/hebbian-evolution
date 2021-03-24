from typing import Dict

import torch as t
import tqdm
from evostrat import NormalPopulation, compute_centered_ranks, normalize
from torch.multiprocessing import Pool, set_start_method, cpu_count
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiplicativeLR

# noinspection PyUnresolvedReferences
import envs
import util
from hebbian_agent import HebbianCarRacingAgent
from meta_agent import MetaAgent

if __name__ == '__main__':
    set_start_method('fork')
    t.multiprocessing.set_sharing_strategy('file_system')

    device = "cuda" if t.cuda.is_available() else "cpu"

    env_args = [
        {},
        # {'side_force': 10.0},
        # {'side_force': -10.0},
        # {'friction': 0.5},
        # {'friction': 2.0}
    ]

    param_shapes = HebbianCarRacingAgent.param_shapes()
    cnn_params = {k: t.randn(s) for k, s in param_shapes.items() if k.startswith('cnn')}
    hebb_shapes = {k: s for k, s in param_shapes.items() if k.startswith('hebb')}


    def constructor(hebb_params: Dict) -> MetaAgent:
        params = dict(**cnn_params, **hebb_params)
        params = {k: p.detach() for k, p in params.items()}
        return MetaAgent([HebbianCarRacingAgent(params, env_arg) for env_arg in env_args])


    population = NormalPopulation(hebb_shapes, constructor, 0.1, True)
    population.param_means = {k: t.randn(shape, requires_grad=True, device=device) for k, shape in hebb_shapes.items()}  # pop mean init hack

    iterations = 300
    pop_size = 200

    optim = SGD(population.parameters(), lr=0.2)
    sched = MultiplicativeLR(optim, lr_lambda=lambda step: 0.995)
    pbar = tqdm.tqdm(range(iterations))
    best_so_far = -1e9
    train_writer, test_writer = util.get_writers('hebbian')

    def fitness_shaping(x):
        return normalize(compute_centered_ranks(x))

    for i in pbar:
        optim.zero_grad()
        with Pool(cpu_count() // 2) as pool:
            raw_fitness = population.fitness_grads(pop_size, pool, compute_centered_ranks)

        train_writer.add_scalar('fitness', raw_fitness.mean(), i)
        train_writer.add_scalar('fitness/std', raw_fitness.std(), i)
        for p_idx, p in enumerate(population.parameters()):
            train_writer.add_histogram('grads/%d' % p_idx, p.grad, i)

        optim.step()
        sched.step()
        population.param_logstds = {k: t.log(t.exp(logstd) * 0.999) for k, logstd in population.param_logstds.items()}  # sigma decay hack
        mean_fit = raw_fitness.mean().item()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (mean_fit, raw_fitness.std().item()))

        all_params = list(cnn_params.values()) + population.parameters()

        if mean_fit > best_so_far:
            best_so_far = mean_fit
            t.save(all_params, 'best.t')
            util.upload_results('best.t')

        if mean_fit > 900:
            t.save(all_params, 'sol.t')
            util.upload_results('sol.t')
            print("Solved.")
            break

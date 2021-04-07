from multiprocessing.pool import Pool
from os import cpu_count
from typing import Dict

import torch as t
import tqdm
from evostrat import NormalPopulation, compute_centered_ranks, normalize
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiplicativeLR

# noinspection PyUnresolvedReferences
import envs
import util
from agents.meta_agent import MetaAgent
from agents.racer.hebbian_racer import HebbianCarRacingAgent
from agents.racer.rnn_racer import RecurrentCarRacingAgent
from agents.racer.static_racer import StaticCarRacingAgent

if __name__ == '__main__':

    device = "cuda" if t.cuda.is_available() else "cpu"

    train_envs = [
        {},
        {'side_force': -10.0},
        {'friction': 0.5},
        {'friction': 2.0}
    ]
    test_env = {'side_force': 10.0}

    agent = RecurrentCarRacingAgent
    param_shapes = agent.param_shapes()


    def constructor(params: Dict) -> MetaAgent:
        params = {k: p.detach() for k, p in params.items()}
        return MetaAgent([agent(params, env_arg) for env_arg in train_envs])


    population = NormalPopulation(param_shapes, constructor, 0.1, True)
    population.param_means = {k: t.randn(shape, requires_grad=True, device=device) for k, shape in param_shapes.items()}  # pop mean init hack

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
            raw_fitness = population.fitness_grads(pop_size, pool, fitness_shaping)

        train_writer.add_scalar('fitness', raw_fitness.mean(), i)
        train_writer.add_scalar('fitness/std', raw_fitness.std(), i)
        for p_idx, p in enumerate(population.parameters()):
            train_writer.add_histogram('grads/%d' % p_idx, p.grad, i)

        optim.step()
        sched.step()
        population.param_logstds = {k: t.log(t.exp(logstd) * 0.999) for k, logstd in population.param_logstds.items()}  # sigma decay hack
        mean_fit = raw_fitness.mean().item()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (mean_fit, raw_fitness.std().item()))

        all_params = population.parameters()

        if mean_fit > best_so_far:
            best_so_far = mean_fit
            t.save(all_params, 'best.t')
            util.upload_results('best.t')

        if mean_fit > 900:
            t.save(all_params, 'sol.t')
            util.upload_results('sol.t')
            print("Solved.")
            break

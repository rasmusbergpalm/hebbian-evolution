from multiprocessing.pool import Pool
from os import cpu_count
from typing import Dict

import torch as t
import tqdm
from evostrat import compute_centered_ranks, normalize, GaussianMixturePopulation
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiplicativeLR

import util
from agents.ant.hebbian_ant import HebbianAnt
from agents.meta_agent import MetaAgent

# noinspection PyUnresolvedReferences
import envs
from agents.racer.hebbian_racer import HebbianCarRacingAgent
from mixed_normal_gmm_population import MixedNormalAndGMMPopulation

if __name__ == '__main__':

    device = "cuda" if t.cuda.is_available() else "cpu"

    ant_train_envs = [
        {'morphology_xml': 'ant.xml'},
        {'morphology_xml': 'ant-long-back.xml'},
        {'morphology_xml': 'ant-damage-left.xml'},
        {'morphology_xml': 'ant-damage-right.xml'},
    ]
    ant_test_env = {'morphology_xml': 'ant-long-front.xml'},
    car_train_envs = [
        {},
        {'side_force': -10.0},
        {'friction': 0.5},
        {'friction': 2.0}
    ]
    car_est_env = {'side_force': 10.0}

    train_envs = car_train_envs
    agent = HebbianCarRacingAgent
    param_shapes = agent.param_shapes()


    def constructor(params: Dict) -> MetaAgent:
        params = {k: p.detach().to("cpu") for k, p in params.items()}
        return MetaAgent([agent(params, env_arg) for env_arg in train_envs])


    rho = 128
    norm_shapes = {k: v for k, v in param_shapes.items() if not k.startswith('hebb')}
    gmm_shapes = {k: v[:-1] for k, v in param_shapes.items() if k.startswith('hebb')}
    n_rules = int(sum([t.Size(s).numel() for s in gmm_shapes.values()]) / rho)
    population = MixedNormalAndGMMPopulation(norm_shapes, gmm_shapes, constructor, 0.1, (n_rules, 5), device)

    # population.mixing_logits = {k: t.randn(ml.shape, requires_grad=True, device=ml.device) for k, ml in population.mixing_logits.items()}

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
        population.gmm_pop.std *= 0.999
        population.normal_pop.std *= 0.999
        mean_fit = raw_fitness.mean().item()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (mean_fit, raw_fitness.std().item()))

        all_params = population.parameters()

        if mean_fit > best_so_far:
            best_so_far = mean_fit
            t.save(all_params, 'best.t')
            util.upload_results('best.t')

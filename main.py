from multiprocessing.pool import Pool
from os import cpu_count
from typing import Dict

import torch as t
import tqdm
from evostrat import compute_centered_ranks, normalize, GaussianMixturePopulation
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiplicativeLR

import util
from agents.ant.hebbian_ant import HebbianAnt
from agents.meta_agent import MetaAgent

# noinspection PyUnresolvedReferences
import envs

if __name__ == '__main__':

    device = "cuda" if t.cuda.is_available() else "cpu"

    train_envs = [
        {'morphology_xml': 'ant.xml'},
        # {'morphology_xml': 'ant-long-back.xml'},
        # {'morphology_xml': 'ant-damage-left.xml'},
        # {'morphology_xml': 'ant-damage-right.xml'},
    ]
    test_env = {'morphology_xml': 'ant-long-front.xml'},

    agent = HebbianAnt
    param_shapes = agent.param_shapes()


    def constructor(params: Dict) -> MetaAgent:
        params = {k: p.detach().to("cpu") for k, p in params.items()}
        return MetaAgent([agent(params, env_arg) for env_arg in train_envs])


    rho = 128
    n_rules = int(sum([t.Size(s).numel() for s in param_shapes.values()]) / rho)
    population = GaussianMixturePopulation({k: t.Size(v[:-1]) for k, v in param_shapes.items()}, (n_rules, 5), constructor, 0.1, device)

    iterations = 500
    pop_size = 500

    optim = Adam(population.parameters(), lr=1.0)
    lr_decay = t.exp(t.log(t.scalar_tensor(0.5)) / 100)  # halves every 100 steps
    sched = MultiplicativeLR(optim, lr_lambda=lambda step: lr_decay)
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
        for k, p in population.mixing_logits.items():
            train_writer.add_histogram("entropy/%s" % k, t.distributions.Categorical(logits=p).entropy(), i)

        means = population.component_means  # (480, 5)
        dist = ((means.unsqueeze(0) - means.unsqueeze(1)) ** 2).sum(dim=2).sqrt()  # (1, 480, 5,) - (480, 1, 5) = (480, 480, 5)
        train_writer.add_histogram("dist", dist, i)
        train_writer.add_image("dist_img", dist, i, dataformats="HW")

        optim.step()
        sched.step()
        population.std *= 0.999
        mean_fit = raw_fitness.mean().item()
        pbar.set_description("avg fit: %.3f, std: %.3f" % (mean_fit, raw_fitness.std().item()))

        all_params = population.parameters()

        if mean_fit > best_so_far:
            best_so_far = mean_fit
            t.save(all_params, 'best.t')
            util.upload_results('best.t')

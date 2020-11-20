import torch as t
from evostrat import NormalPopulation

# noinspection PyUnresolvedReferences
import envs
from hebbian_agent import HebbianCarRacingAgent
from meta_agent import MetaAgent
from static_car import StaticCarRacingAgent

if __name__ == '__main__':
    envs = [
        {},
        {'side_force': 10.0},
        {'side_force': -10.0},
        {'friction': 0.5},
        {'friction': 2.0}
    ]

    env = envs[4]
    agent = HebbianCarRacingAgent
    params = t.load('results_c89a28c_best.t')


    def constructor(params) -> MetaAgent:
        return agent.from_params(params, env)


    agent = agent(env)
    shapes = {k: p.shape for k, p in agent.get_params().items()}
    population = NormalPopulation(shapes, constructor, std=0.000001)

    population.param_means = {k: p for k, p in zip(agent.get_params().keys(), params)}
    inds, logps = zip(*population.sample(2))

    inds[0].fitness(render=True)

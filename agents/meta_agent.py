from typing import Sequence

from evostrat import Individual


class MetaAgent(Individual):

    def __init__(self, agents: Sequence[Individual]):
        self.agents = agents

    def fitness(self) -> float:
        return sum([agent.fitness() for agent in self.agents]) / len(self.agents)

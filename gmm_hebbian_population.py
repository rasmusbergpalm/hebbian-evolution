from typing import Dict, Iterable

import torch as t
import torch.distributions as d

from es import Population
from hebbian_agent import HebbianAgent


class GMMHebbianPopulation(Population):
    def __init__(self, num_learning_rules: int, scale=float):
        mixing_logits_tensors = {k: t.tensor(1000*t.randn(v.shape[:-1] + (num_learning_rules,)), requires_grad=False) for k, v in HebbianAgent().get_params().items()}
        learning_rule_cluster_means = t.randn((num_learning_rules, 5), requires_grad=True)
        self.mixing_logits_tensors = mixing_logits_tensors
        self.learning_rule_cluster_means = learning_rule_cluster_means.sg(("M", 5))
        self.scale = scale

    def average_mixing_entroy(self):
        return {key: d.Categorical(logits=mixing_logits_tensor).entropy().mean().item() for key, mixing_logits_tensor in self.mixing_logits_tensors.items()}

    def parameters(self) -> Iterable[t.Tensor]:
        return [self.learning_rule_cluster_means]

    def sample(self, n) -> Iterable[HebbianAgent]:
        return [
            HebbianAgent.from_params({
                key: d.Normal(loc=self.learning_rule_cluster_means[d.Categorical(logits=mixing_logits_tensor).sample()], scale=self.scale).sample()
                for key, mixing_logits_tensor
                in self.mixing_logits_tensors.items()
            })
            for _ in range(n)
        ]

    def log_prob(self, individual: HebbianAgent) -> float:
        log_p_h = 0.0

        for key, h in individual.get_params().items():
            mixing_logits_tensor = self.mixing_logits_tensors[key]
            n_in, n_out, M = mixing_logits_tensor.shape
            log_p_k = t.log_softmax(mixing_logits_tensor, dim=2).unsqueeze(-1).sg((n_in, n_out, "M", 1))

            p_h_given_k = d.Normal(loc=self.learning_rule_cluster_means, scale=self.scale).expand((n_in, n_out, M, 5)).sg((n_in, n_out, "M", 5))
            log_p_h_given_k = p_h_given_k.log_prob(h.unsqueeze(2)).sg((n_in, n_out, "M", 5))

            log_p_h += t.logsumexp((log_p_k + log_p_h_given_k).sum(dim=-1).sg((n_in, n_out, "M")), dim=-1).sg((n_in, n_out)).sum()

        return log_p_h

    def save(self, fname):
        t.save({"mixing_logits_tensors": self.mixing_logits_tensors, "learning_rule_cluster_means": self.learning_rule_cluster_means}, fname)

    def load(self, fname):
        d = t.load(fname)
        self.mixing_logits_tensors = d["mixing_logits_tensors"]
        self.learning_rule_cluster_means = d["learning_rule_cluster_means"]

"""
networks.py — the shared policy network (the CTDE critic will be added for F2).

Actor: local observation -> action, as a Gaussian policy.
  - forward(obs)        -> action MEAN. Used directly for imitation learning
                           (plain regression) and for deterministic evaluation.
  - distribution(obs)   -> Normal(mean, exp(log_std)) for PPO sampling / log-probs.

A state-independent log_std parameter makes it a valid stochastic policy while
keeping IL as simple MSE regression on the mean. The same class serves
Formulation 1 (act_dim=1, lambda) and full-force / residual (act_dim=3).
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim=24, act_dim=1, hidden=(128, 128), log_std_init=-0.5):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.mean = nn.Linear(last, act_dim)
        # state-independent log std (standard for continuous PPO)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def forward(self, obs):
        """Action mean — used for IL regression and deterministic eval."""
        return self.mean(self.body(obs))

    def distribution(self, obs):
        """Gaussian policy for RL: Normal(mean, exp(log_std))."""
        return torch.distributions.Normal(self.forward(obs), self.log_std.exp())

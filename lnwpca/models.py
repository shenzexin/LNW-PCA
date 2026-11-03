"""EdgeConv reliability model and deterministic actor-critic networks."""
import torch
from torch import nn


class EdgeConv(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * inputs, outputs), nn.ReLU(),
                                 nn.Linear(outputs, outputs), nn.ReLU())

    def forward(self, x, graph):
        batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
        neighbors = x[batch, graph]
        center = x[:, :, None].expand_as(neighbors)
        return self.mlp(torch.cat((center, neighbors - center), -1)).amax(2)


class ReliabilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge1, self.edge2 = EdgeConv(8, 64), EdgeConv(64, 128)
        self.se = nn.Sequential(nn.Linear(128, 8), nn.ReLU(), nn.Linear(8, 128), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features, graph):
        x = self.edge2(self.edge1(features, graph), graph)
        x = x * self.se(x.mean(1))[:, None]
        return self.head(x).squeeze(-1)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128),
                                     nn.ReLU(), nn.Linear(128, 2), nn.Tanh())

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128),
                                     nn.ReLU(), nn.Linear(128, 1))

    def forward(self, state, action):
        return self.network(torch.cat((state, action), -1)).squeeze(-1)

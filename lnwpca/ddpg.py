"""Replay-based deterministic policy gradient updates."""
from collections import deque
from copy import deepcopy
import random
import numpy as np
import torch
from .models import Actor, Critic


class DDPG:
    def __init__(self, capacity=100000, seed=42):
        torch.manual_seed(seed)
        self.random = random.Random(seed)
        self.actor, self.critic = Actor(), Critic()
        self.target_actor, self.target_critic = deepcopy(self.actor), deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay = deque(maxlen=capacity)

    def remember(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))

    def update(self, batch_size=128):
        if len(self.replay) < batch_size:
            return None
        batch = self.random.sample(list(self.replay), batch_size)
        state, action, reward, next_state, done = [
            torch.tensor(np.asarray(x), dtype=torch.float32) for x in zip(*batch)]
        with torch.no_grad():
            target = reward + .99 * (1 - done) * self.target_critic(
                next_state, self.target_actor(next_state))
        critic_loss = (self.critic(state, action) - target).square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            for target_net, net in [(self.target_actor, self.actor), (self.target_critic, self.critic)]:
                for target_parameter, parameter in zip(target_net.parameters(), net.parameters()):
                    target_parameter.mul_(.995).add_(parameter, alpha=.005)
        return float(actor_loss.detach()), float(critic_loss.detach())


def reward(mean_residual, rotation_error, translation_error, inlier_ratio, occlusion, diagonal):
    if diagonal <= 0:
        raise ValueError('Reference diagonal must be positive.')
    return (-100 / diagonal * mean_residual - 18 / np.pi * rotation_error
            - 20 / diagonal * translation_error + .5 * inlier_ratio - .5 * occlusion)

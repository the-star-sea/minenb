import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DiscriminatorNN(nn.Module):

    def __init__(self, sa_dim):
        super(DiscriminatorNN, self).__init__()
        self.hidden1 = nn.Linear(sa_dim, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.output(x)


class Discriminator():

    def __init__(self, state_dim, action_dim):
        self.disc = DiscriminatorNN(state_dim + action_dim)

    def update(self, expert_states, expert_actions, policy_batch,disc_optim):
        loss = 0.0
        policy_states, policy_actions, _, _, _ = map(torch.FloatTensor, map(np.stack, zip(*policy_batch)))

        expert_d = self.disc(torch.cat([expert_states, expert_actions], dim=1))
        policy_d = self.disc(torch.cat([policy_states, policy_actions], dim=1))

        expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()))
        policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()))

        gail_loss = expert_loss + policy_loss

        disc_optim.zero_grad()
        gail_loss.backward()
        disc_optim.step()

        loss += gail_loss.item()




    def predict_rewards(self, rollout):
        states, actions, logprob_base, rewards, dones = map(np.stack, zip(*rollout))
        with torch.no_grad():
            policy_mix = torch.cat([torch.FloatTensor(states), torch.FloatTensor(actions)], dim=1)
            policy_d = self.disc(policy_mix).squeeze()
            score = torch.sigmoid(policy_d)
            # gail_rewards = - (1-score).log()
            gail_rewards = score.log() - (1 - score).log()
            return (states, actions, logprob_base, gail_rewards.numpy(), dones)
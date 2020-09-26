import torch
from model.Network import DenseNet
from torch import nn
from torch.optim import Adam

actor = DenseNet(5, 2, hidden_activate=nn.ReLU())
critic = DenseNet(7, 1, hidden_activate=nn.ReLU())

class actor_critic(nn.Module):
    def __init__(self, actor, critic):
        super(actor_critic, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, obs):
        a = self.actor(obs)
        input = torch.cat((obs, a), axis=-1)
        Q = self.critic(input)
        return Q

actor_optim = Adam(actor.parameters(), lr=1e-1)
critic_optim = Adam(critic.parameters(), lr=1e-1)

input = torch.rand(10, 5)
tgt = torch.rand(10, 1)
loss_fun = torch.nn.MSELoss()

a = actor(input)
innn = torch.cat((input, a), axis=-1)
b = critic(innn)

actor.zero_grad()
torch.mean(b).backward()
actor_optim.step()

ab = actor(input)
bb = critic(innn)

totalmodel = actor_critic(actor, critic)
totalmodel(input)
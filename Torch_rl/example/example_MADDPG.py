import gym
import time
import torch
from Torch_rl.agent.DDPG_2 import DDPG_Agent
from Torch_rl.model.Network import DenseNet
from torch import nn
#%%
envID = "Pendulum-v0"
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M', time.localtime())
path = "savedate" + '/' + envID + "-ddpg-" + nowtime+'/'


class actor(nn.Module):
    "the input and output of the actor share the same dim"
    "input dim: batch, actor_num, observation_shape "
    "output dim: batch, actor_num, action_shape "
    def __init__(self, num_actor, input_size, output_size, hidden_layer=[64, 64],
                       hidden_activate=nn.ReLU(), output_activate=None,
                       BatchNorm = False):
        self.actor = []
        self.num_actor = num_actor
        for i in range(num_actor):
            self.actor.append(DenseNet(input_size, output_size, hidden_layer=hidden_layer,
                           hidden_activate=hidden_activate, output_activate=output_activate,
                           BatchNorm = BatchNorm))
        self.gpu = False

    def forward(self, obs):
        if self.gpu:
            obs = obs.cuda(self.device)
        a,b,c = obs.shape()
        assert b == len(self.actor)
        action = []
        for i in range(self.num_actor):
            action.append(self.actor[i].forward(obs[:, i, :]))
        action = torch.cat(action, dim=1)
        return action

    def to_gpu(self, device=None):
        for i in range(self.num_actor):
            self.actor[i].to_gpu(device)
        self.gpu = True
        self.device = device


class critic(nn.Module):
    "the input and output of the actor share the same dim"
    "input observation dim: batch, actor_num, observation_shape "
    "input action dim: batch, actor_num, action_shape "
    def __init__(self, num_actor, input_size, output_size, hidden_layer=[64, 64],
                       hidden_activate=nn.ReLU(), output_activate=None,
                       BatchNorm = False):
        self.critic = []
        self.num_actor = num_actor
        self.critic = DenseNet(input_size*num_actor, output_size, hidden_layer=hidden_layer,
                           hidden_activate=hidden_activate, output_activate=output_activate,
                           BatchNorm = BatchNorm)
        self.gpu = False

    def forward(self, obs, action):
        if self.gpu:
            obs = obs.cuda(self.device)
            action = action.cuda(self.device)
        a1, b1, c1 = obs.shape()
        a2, b2, c2 = action.shape()
        assert b1 == self.num_actor and b2 == self.num_actor and a1 == a2
        obs_combin = obs.view(a1, -1)
        ac_combin = obs.view(a2, -1)
        input = torch.cat((obs_combin,ac_combin), dim = -1)
        Q = self.critic.forward(input)
        return Q

    def to_gpu(self, device=None):
        self.critic.to_gpu(device=device)
        self.gpu = True
        self.device = device


num_actor = 2
'The observation_space and action_space is the single agent observation '

actor = actor(num_actor, env.observation_space.shape[0], env.action_space.shape[0],
                 hidden_activate=nn.ReLU(), hidden_layer=[64, 64])
critic = critic(num_actor, env.observation_space.shape[0]+env.action_space.shape[0], 1,
                  hidden_activate=nn.ReLU(), hidden_layer=[64, 64])
Agent = DDPG_Agent(env, actor, critic, gamma=0.99, actor_lr=1e-4, critic_lr=1e-4, path=path, sperate_critic=True)

Agent.train(max_step=100000, render=False, verbose=2)
Agent.test(max_step=10000, render=True, verbose=2)



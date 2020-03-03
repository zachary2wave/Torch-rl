import gym
import time
from Torch_rl.agent.DQN import DQN_Agent
from Torch_rl.model.Network import DenseNet
from torch import nn
from Torch_rl.common.Policy_for_DQN import BoltzmannQPolicy
#%%
envID = 'D_place_action-v0'
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M', time.localtime())
path = "savedate" + '/' + envID + "-DQN-" + nowtime+'/'
#%%

actor = DenseNet(env.observation_space.shape[0], env.action_space.shape[0], hidden_activate=nn.Tanh())
critic = DenseNet(env.observation_space.shape[0]+env.action_space.shape[0], 1, hidden_activate=nn.Tanh())
Agent = DQN_Agent(env, actor, critic, gamma=0.99, path=path)

Agent.train(max_step=10000, render=True, verbose=2)
Agent.test(max_step=10000, render=True, verbose=2)



import gym
import time
from agent.HIRO import HIRO_Agent
from model.Network import DenseNet
from torch import nn
from common.Policy_for_DQN import BoltzmannQPolicy
#%%
envID = 'D_place_action-v0'
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M', time.localtime())
path = "savedate" + '/' + envID + "-DQN-" + nowtime+'/'
#%%
goal = gym.spaces.Box(low=-1, high=1, shape=(5,))
H_model = DenseNet(env.observation_space.shape[0], 32, hidden_activate=nn.Tanh())
L_model = DenseNet(env.observation_space.shape[0]+goal.shape[0], env.action_space.n, hidden_activate=nn.Tanh())
Agent = HIRO_Agent(env, "DDPG", H_model, "DQN", L_model,goal=goal, gamma=0.99, path=path)

Agent.train(max_step=10000, render=True, verbose=2)
Agent.test(max_step=10000, render=True, verbose=2)



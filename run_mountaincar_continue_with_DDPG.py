import gym
import time
from agent.ddpg import DDPG_Agent
from model.Network import DenseNet
from torch import nn
from common.Policy_for_DQN import BoltzmannQPolicy
#%%
envID = "MountainCarContinuous-v0"
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M',time.localtime())
path = "savedate" + '/' + envID + "dqn" + nowtime+'/'
#%%
policy = BoltzmannQPolicy()
actor = DenseNet(env.observation_space.shape[0], env.action_space.shape[0], hidden_activate=nn.Tanh())
critic = DenseNet(env.observation_space.shape[0]+env.action_space.shape[0], 1, hidden_activate=nn.Tanh())
Agent = DDPG_Agent(env, actor, critic, policy, gamma=0.99, path=path)

Agent.train(max_step=100000, render=True, verbose=2)
Agent.test(max_step=10000, render=True, verbose=2)

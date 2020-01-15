import gym
import time
from agent.DQN import DQN_Agent
from model.Network import DenseNet
from torch import nn
from common.Policy_for_DQN import EpsGreedyQPolicy
#%%
envID = "MountainCar-v0"
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M',time.localtime())
path = "savedate" + '/' + envID + "dqn" + nowtime+'/'
#%%

policy = EpsGreedyQPolicy()
model = DenseNet(env.observation_space.shape[0], env.action_space.n, hidden_activate=nn.Tanh())

Agent = DQN_Agent(env, model, policy, gamma=0.90, lr=1e-3, path=path)

Agent.train(max_step=100000, render=True, verbose=2)
Agent.save_weights(path)
Agent.test(max_step=10000, render=True, verbose=2)

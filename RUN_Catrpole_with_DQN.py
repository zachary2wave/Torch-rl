import gym
import time
from agent.DQN import DQN_Agent
from model.Network import DenseNet
from torch import nn
from common.Policy_for_DQN import BoltzmannQPolicy
#%%
envID = "CartPole-v0"
env = gym.make(envID)
nowtime = time.strftime('%y%m%d%H',time.localtime())
path = "savedate" + '/' + envID + "dqn" + nowtime+'/'
#%%

policy = BoltzmannQPolicy()
model = DenseNet(env.observation_space.shape[0], env.action_space.n, hidden_activate=nn.Tanh())

Agent = DQN_Agent(env, model, policy, path=path)

Agent.train(max_step=50000, render=False, verbose=2)
Agent.test(max_step=50000, render=False, verbose=2)

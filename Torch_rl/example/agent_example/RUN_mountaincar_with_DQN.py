import gym
import time
from Torch_rl import DQN
from Torch_rl.model.Network import DenseNet
from torch import nn
from Torch_rl.common.Policy_for_DQN import EpsGreedyQPolicy
#%%
envID = "MountainCar-v0"
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M',time.localtime())
path = "savedate" + '/' + envID + "-dqn-" + nowtime+'/'
#%%

policy = EpsGreedyQPolicy()
model = DenseNet(env.observation_space.shape[0], env.action_space.n, hidden_activate=nn.Tanh())

Agent = DQN(env, model, policy, gamma=0.90, lr=1e-3, path=path)

Agent.train(max_step=100000, render=False, verbose=2)
Agent.save_weights(path)
Agent.test(max_step=10000, render=True, verbose=2)

import gym
import time
from Torch_rl import TD3
from Torch_rl.model.Network import DenseNet
from torch import nn
#%%
envID = "Pendulum-v0"
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M', time.localtime())
path = "savedate" + '/' + envID + "-ddpg-" + nowtime+'/'
#%%
actor = DenseNet(env.observation_space.shape[0], env.action_space.shape[0],
                 hidden_activate=nn.ReLU(), hidden_layer=[64, 64])
critic = DenseNet(env.observation_space.shape[0]+env.action_space.shape[0], 1,
                  hidden_activate=nn.ReLU(), hidden_layer=[64, 64])
Agent = TD3(env, actor, critic, gamma=0.99, path=path)

Agent.train(max_step=50000, render=False, verbose=2)
Agent.test(max_step=10000, render=True, verbose=2)



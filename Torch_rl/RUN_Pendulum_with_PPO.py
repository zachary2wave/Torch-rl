import gym
import time
from Torch_rl import PPO
from Torch_rl.model.Network import DenseNet
from torch import nn

#%%
envID ="Pendulum-v0"
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M',time.localtime())
path = "savedate" + '/' + envID + "dqn" + nowtime+'/'
#%%
policy_model = DenseNet(env.observation_space.shape[0], env.action_space.shape[0]*2,
                 hidden_activate=nn.ReLU(), hidden_layer=[64, 64])
value_model = DenseNet(env.observation_space.shape[0], 1,
                  hidden_activate=nn.ReLU(), hidden_layer=[64, 64])

Agent = PPO(env, policy_model, value_model, path=path)

Agent.train(max_step=150000, render=False, verbose=2)
Agent.test(max_step=10000, render=True, verbose=2)

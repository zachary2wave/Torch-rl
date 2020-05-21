import gym
import time
from Torch_rl import PPO
from Torch_rl.model.Network import DenseNet, LSTM_Dense_Hin
from torch import nn

#%%
envID ="Pendulum-v0"
env = gym.make(envID)

nowtime = time.strftime('%y%m%d%H%M',time.localtime())
path = "../savedate" + '/' + envID + "ppo" + nowtime+'/'
#%%
policy_model = LSTM_Dense_Hin(env.observation_space.shape[0], env.action_space.shape[0],lstm_layer=1,lstm_unit=64,
                 hidden_activate=nn.Tanh(), dense_layer=[64])
value_model = DenseNet(env.observation_space.shape[0], 1,
                  hidden_activate=nn.Tanh(), hidden_layer=[64, 64])

Agent = PPO(env, policy_model, value_model, gamma=0.90,
            lr=1e-4, running_step=2048, batch_size=64, value_train_round=10, path=path, lstm_enable=True)
Agent.cuda()
Agent.train(max_step=1500000, render=False, verbose=0, record_ep_inter=1)
Agent.test(max_step=10000, render=True, verbose=2)

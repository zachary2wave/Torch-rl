import torch
import numpy as np
from Torch_rl.common.memory import ReplayMemory
from Torch_rl.agent.core_value import Agent_value_based
from copy import deepcopy
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from Torch_rl.common.loss import huber_loss
from torch.autograd import Variable
from gym import spaces as Space
from Torch_rl.common.Policy_for_DQN import BoltzmannQPolicy

class HIRO_Agent(Agent_value_based):
    def __init__(self, env,
                 H_policy, H_model, L_policy, L_model,
                 goal = Space.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                 # step for H_model
                 step_interval = 10, H_train_interval=100, H_train_time = 100,
                 ## hyper-parameter
                 gamma=0.90, H_lr=1e-3, L_lr = 1e-3, batch_size=32, buffer_size=50000, learning_starts=1000,
                 H_target_network_update_freq=500, L_target_network_update_freq=500,
                 decay=False, decay_rate=0.9,
                 ## prioritized_replay
                 ##
                 path=None):
        """

        :param env:
        :param H_policy:
        :param H_model:
        :param L_policy:
        :param L_model:
        :param goal:
        :param step_interval:
        :param gamma:
        :param H_lr:
        :param L_lr:
        :param batch_size:
        :param buffer_size:
        :param learning_starts:
        :param H_target_network_update_freq:
        :param L_target_network_update_freq:
        :param decay:
        :param decay_rate:
        :param path:
        """

        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.step_interval = step_interval

        # self.replay_buffer = ReplayMemory(buffer_size)
        # generate policy
        if H_policy == "DDPG" and isinstance(goal, Space.Box) and len(H_model) == 2:
            from agent.DDPG import DDPG_Agent
            if isinstance(H_lr,list):
                ac_lr = H_lr[0]
                cr_lr = H_lr[1]
            else:
                ac_lr = H_lr
                cr_lr = H_lr
            if isinstance(H_target_network_update_freq,list):
                actor_target_network_update_freq = H_target_network_update_freq[0]
                critic_target_network_update_freq = H_target_network_update_freq[1]
            else:
                actor_target_network_update_freq = H_target_network_update_freq
                critic_target_network_update_freq = H_target_network_update_freq
            self.H_agent = DDPG_Agent(env, H_model[0], H_model[1],
            actor_lr=ac_lr, critic_lr=cr_lr,
            actor_target_network_update_freq=actor_target_network_update_freq,
            critic_target_network_update_freq=critic_target_network_update_freq,
            ## hyper-parameter
            gamma=gamma, batch_size=batch_size, buffer_size=buffer_size, learning_starts=learning_starts,
            ## decay
            decay=decay, decay_rate=decay_rate,
            )
            self.H_main_net = self.H_agent.actor

        if H_policy == "PPO" and isinstance(goal, Space.Box):
            from agent.PPO import PPO_Agent
            self.high_agent = PPO_Agent()

        if L_policy == "DQN":
            from agent.DQN import DQN_Agent
            self.L_agent = DQN_Agent(env, L_model, BoltzmannQPolicy,
                 gamma=gamma, lr=L_lr, batch_size=batch_size, buffer_size=buffer_size, learning_starts=learning_starts,
                 target_network_update_freq=L_target_network_update_freq,
                 decay=decay, decay_rate=decay_rate,
                 double_dqn=True, dueling_dqn=False, dueling_way="native")
            self.L_main_net = self.L_agent.Q_net

    def forward(self, observation):
        observation = observation.astype(np.float32)
        observation = torch.from_numpy(observation)
        if self.step % self.step_interval == 0:
            goal = self.high_agent.forward(observation)
        if isinstance(goal,tuple):
            self.goal, Q = goal[0], goal[1]
        else:
            self.goal = goal
        L_observation = torch.cat(inputs=(observation, self.goal), dimension=0)
        action = self.L_agent.forward(L_observation)
        if isinstance(action, tuple):
            action, Q = action[0], action[1]
        else:
            action = action



        return action

    def backward(self, sample_):
        self.L_agent.backward(sample_)
        if self.step % self.step_interval == 0:
            self.L_agent.replay_buffer.sample(self.batch_size)










    def load_weights(self, filepath):
        pass

    def save_weights(self, filepath, overwrite=False):
        pass


import torch
import numpy as np
from common.memory import ReplayMemory
from agent.core import Agent
from copy import deepcopy
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from common.loss import huber_loss
from torch.autograd import Variable

class HIRO_Agent(Agent):
    def __init__(self, env, H_model, L_model,
                 # step for H_model
                 delay_step = 10,
                 ## hyper-parameter
                 gamma=0.90, H_lr=1e-3, L_lr = 1e-3, batch_size=32, buffer_size=50000, learning_starts=1000,
                 target_network_update_freq=500,
                 ## prioritized_replay
                 ##
                 path=None):
        """

        :param env:
        :param H_model:
        :param L_model:
        :param delay_step:
        :param gamma:
        :param H_lr:
        :param L_lr:
        :param batch_size:
        :param buffer_size:
        :param learning_starts:
        :param target_network_update_freq:
        :param path:
        """

        self.env = env

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq

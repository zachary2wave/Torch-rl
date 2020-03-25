import torch
import numpy as np
from Torch_rl.common.memory import ReplayMemory
from Torch_rl.agent.core import Agent
from copy import deepcopy
from torch.optim import Adam
from torch import nn
from Torch_rl.common.loss import huber_loss
from torch.autograd import Variable


class GAIL_Agent(Agent):
    def __init__(self, env, Policy_model,  adversary_model, expert_model=None,
                 Adversary_lr=1e-4, Policy_lr=1e-3,
                 actor_target_network_update_freq=1000, critic_target_network_update_freq=1000,
                 actor_training_freq=1, critic_training_freq=1,
                 ## hyper-parameter
                 gamma=0.99, batch_size=32, buffer_size=50000, learning_starts=1000,
                 ## lr_decay
                 decay=False, decay_rate=0.9, critic_l2_reg=1e-2, clip_norm=None,
                 ##
                 path=None):
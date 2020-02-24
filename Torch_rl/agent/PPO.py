import torch
from Torch_rl.agent.core import Agent
from Torch_rl.common.memory import Sequence_Replay_Memory
from copy import deepcopy
from Torch_rl.common.distribution import *


class PPO_Agent(Agent):

    def __init__(self, env, policy_model, value_model,
                 lr=1e-5, updata_time = 1,
                 ## hyper-parameter
                 gamma=0.90, batch_size=32, buffer_size=50000, learning_starts=1000,
                 ## decay
                 decay=False, decay_rate=0.9,
                 ##
                 path=None):
        if value_model is None:
            if value_model == "shared":
                self.value_model = policy_model
            elif value_model == "copy":
                self.value_model = deepcopy(policy_model)
            else:
                self.value_model = value_model

        self.pdtype = make_pdtype(env.action_space)




        super(PPO_Agent,self).__init__(path)


    def forward(self, observation):




    def backward(self, sample):
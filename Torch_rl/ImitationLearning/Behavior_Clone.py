import torch
import numpy as np
from Torch_rl.ImitationLearning.core_IL import Agent_IL
from copy import deepcopy
from torch import nn
from Torch_rl.common import logger
from Torch_rl.common.memory import ReplayMemory
from torch.optim import Adam

class BC_Agent(Agent_IL):

    def __init__(self, env, base_algorithm, policy_network, value_network = None,
                 batch_size=32, lr=1e-4,
                 path=None):
        self.env = env
        self.base_algorithm = base_algorithm
        self.policy_network = policy_network
        self.value_network = value_network
        self.batch_size = batch_size

        self.loss_cal = nn.MSELoss()
        self.policy_model_optim = Adam(self.policy_network.parameters(), lr=lr)
        if self.value_network is not None:
            self.value_model_optim = Adam(self.value_network.parameters(), lr=lr)

        super(BC_Agent, self).__init__(path)

    def training_with_data(self, expert_data, max_imitation_learning_step, training_ways):

        self.step = 0

        while self.step < max_imitation_learning_step:
            if training_ways == "random":
                samples = expert_data.sample(self.batch_size)
            elif training_ways == "episode":
                samples = expert_data.sample_episode()
            elif training_ways == "fragment":
                samples = expert_data.sample_fragment(self.batch_size)

            actions = self.policy_network.forward(samples["s"])
            loss = self.loss_cal(actions, samples["a"])
            self.policy_model_optim.zero_grad()
            loss.backward()
            self.policy_model_optim.step()

    def training_with_policy(self, expert_policy, max_imitation_learning_step=1e5,
                            max_ep_cycle=2000, buffer_size=32):
        self.step = 0
        s = self.env.reset()
        loss_BC = 0
        ep_step, ep_reward, ep_loss = 0, 0, 0
        expert_action_set,policy_action_set = [],[]

        for _ in range(max_imitation_learning_step):
            self.step += 1
            ep_step += 1
            a_expert = expert_policy(s)
            a_policy = self.policy_network.forward(s)

            expert_action_set.append(torch.tensor(a_expert))
            policy_action_set.append(a_policy)
            s_, r, done, info = self.env.step(a_policy)
            ep_reward += r
            sample = {"s": s, "a": a_policy, "a_expert":a_expert, "s_": s_, "r": r, "tr": done}
            s = s_[:]

            if len(policy_action_set) > buffer_size:

                loss = self.loss_cal(expert_action_set, policy_action_set)
                ep_loss += loss.cpu().detach().numpy()
                self.policy_model_optim.zero_grad()
                loss.backward()
                self.policy_model_optim.step()

            if done or ep_step>max_ep_cycle:
                ep_step = 0
                logger.record_tabular("steps", self.step)
                logger.record_tabular("loss", ep_loss)
                logger.record_tabular("loss", ep_reward)







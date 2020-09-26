import torch
import numpy as np
from Torch_rl.ImitationLearning.core_IL import Agent_IL
from copy import deepcopy
from torch.optim import Adam
from torch import nn
from Torch_rl.common.memory import ReplayMemory
from types import MethodType,FunctionType


class GAIL_Agent(Agent_IL):
    def __init__(self, env, base_algorithm, adversary_model, policy_network, value_network = None,
                 Adversary_lr=1e-4, ent_coeff = 1e-3, batch_size=32,
                 ##
                 path=None):

        self.env = env

        self.policy_network = policy_network
        self.value_network = value_network
        self.dist = base_algorithm.dist
        self.base_algorithm = base_algorithm
        self.adversary_model = adversary_model
        self.adversary_model_optim = Adam(self.adversary_model.parameters(), lr=Adversary_lr)

        self.entcoeff = ent_coeff
        self.batch_size = batch_size
        self.loss_calculator = nn.CrossEntropyLoss()


        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss"]
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss"]

    def training_with_data(self, expert_data, max_imitation_learning_episode, training_ways):

        self.episode = 0

        while self.step < max_imitation_learning_episode:
            if training_ways == "random":
                samples = expert_data.sample(self.batch_size)
            elif training_ways == "episode":
                samples = expert_data.sample_episode()
            elif training_ways == "fragment":
                samples = expert_data.sample_fragment(self.batch_size)
            self.episode +=1
            expert_action = samples["a"]
            generator_action = self.policy_network.forward(samples["s"])
            if self.value_network is not None:
                Q = self.value_network.forward(samples["s"])
            if self.gpu:
                expert_action.cuda()
                generator_action.cuda()
                for key in samples.keys():
                    samples[key] = samples[key].cuda()

            IL_reward = self.Discriminator_training(samples, expert_action, generator_action)
            # for flag,rew in enumerate(IL_reward):
            #     sample_new = {"s": samples["s"][flag], "a": generator_action, "s_": samples["s_"][flag], "r": rew, "tr": samples["tr"][flag]}
            samples["r"] = IL_reward
            samples["value"] = Q
            samples["logp"] = -1.9189 * np.ones_like(IL_reward)

            self.base_algorithm.backward(samples)



    def training_with_policy(self, expert_policy, max_imitation_learning_step):

        self.step = 0
        s = self.env.reset()
        buffer = ReplayMemory(self.batch_size, ["value", "logp"])
        expert_action_set,generator_action_set=[],[]
        while self.step < max_imitation_learning_step:
            expert_action = expert_policy(s)
            generator_action = self.policy_network.forward(s)
            s_, r, done, info = self.env.step(generator_action.cpu().squeeze(0).numpy())
            Q = self.value_network.forward(s)
            IL_reward = self.Discriminator_training(s, expert_action, generator_action)
            sample_ = {
                "s": s,
                "a": generator_action.squeeze(0),
                "r": IL_reward,
                "tr": torch.tensor([int(done)]),
                "s_":torch.from_numpy(s_),
                "logp": -1.9189,
                "value": Q}

            buffer.push(sample_)
            # expert_action_set.append(expert_action)
            # generator_action_set.append(generator_action)

            if self.step % self.batch_size==0 and self.step>1:
                self.base_algorithm.update(buffer.memory)




    def Discriminator_training(self,sample, expert_action, generator_action):
        expert_input = torch.cat((sample["s"],expert_action), dim=1)
        advertise_judgement = self.adversary_model.forward(expert_input)
        expert_acc = self.loss_cal(advertise_judgement, torch.ones_like(advertise_judgement))

        generator_input = torch.cat((sample["s"], generator_action), dim=1)
        generator_judgement = self.adversary_model.forward(generator_input)
        generator_acc = self.loss_calculator(generator_judgement, torch.zeros_like(generator_judgement))

        logits = torch.cat([advertise_judgement, generator_judgement], dim = 1)
        entropy = - logits*torch.log(logits) -(1-logits)*torch.log(1-logits)

        entropy_loss = - self.entcoeff * entropy
        total_loss = expert_acc + generator_acc - self.entcoeff * entropy
        self.adversary_model_optim.zero_grad()
        total_loss.backward()
        self.adversary_model_optim.step()
        IL_reward = -torch.log(1 - generator_judgement + 1e-8)

        return IL_reward



    def cuda(self):
        self.policy_network.to_gpu()
        self.value_network.to_gpu()
        self.adversary_model.to_gpu()
        self.loss_cal = self.loss_cal.cuda()
        self.gpu = True






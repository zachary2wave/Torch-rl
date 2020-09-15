import torch
import numpy as np
from Torch_rl.common.memory import ReplayMemory
from Torch_rl.agent.PPO import PPO_Agent
from copy import deepcopy
from torch.optim import Adam
from torch import nn
from Torch_rl.common.loss import huber_loss
from torch.autograd import Variable
from Torch_rl.common import logger
from types import MethodType,FunctionType


class GAIL_Agent(PPO_Agent):
    def __init__(self, env, Policy_model, Value_model, adversary_model, expert_guidence=None,
                 Adversary_lr=1e-4, Policy_lr=1e-3, ent_coef=0.01, Generative_Adversarial_Time = 10000,
                 ## hyper-parameter
                 gamma=0.99, batch_size=32, lam=0.95, cliprange=0.2, vf_coef=0.5,value_train_round=10,
                 running_step=2048, running_ep=20, value_regular=0.01,
                 ## lr_decay
                 decay=False, decay_rate=0.9, critic_l2_reg=1e-2, clip_norm=None,
                 ##
                 path=None):

        self.adversary_model = adversary_model
        self.adversary_lr = Adversary_lr
        self.GA_time = Generative_Adversarial_Time
        if isinstance(expert_guidence, FunctionType) or isinstance(expert_guidence, MethodType):
            self.expert_guidence = expert_guidence
            self.IL_way = "guide_by_policy"
        else:
            self.expert_data = expert_guidence
            self.IL_way = "guide_by_data"


        super(PPO_Agent, self).__init__(env=env, policy_model=Policy_model, value_model=Value_model,
                 lr=Policy_lr, ent_coef=ent_coef, vf_coef=vf_coef,
                 ## hyper-parawmeter
                 gamma=gamma, lam=lam, cliprange=cliprange, batch_size=batch_size, value_train_round=value_train_round,
                 running_step=running_step, running_ep=running_ep, value_regular=value_regular,
                 ## decay
                 decay=decay, decay_rate=decay_rate, lstm_enable=False,
                 ##
                 path=path)


        self.loss_calculator = nn.CrossEntropyLoss()


        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss"]
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss"]


    def imitation_learning(self,max_step=50000, max_ep_cycle = 2000):




        
        if self.IL_way == "guide_by_data":
            sample = self.expert_data.sample(self.batch_size)
            expert_action=sample["a"]
            generator_action = self.policy.forward(sample["s"])
        else:
            sample = deepcopy(sample_input)
            expert_action = self.expert_guidence(sample["s"])
            generator_action = sample["a"]

        if self.gpu:
            expert_action.cuda()
            generator_action.cuda()
            for key in sample.keys():
                sample[key] = sample[key].cuda()

    def Discriminator_training(self,sample,expert_action,generator_action):
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

        IL_reward = -torch.log(1 - generator_judgement + 1e-8)



    def cuda(self):
        self.policy.to_gpu()
        self.value.to_gpu()
        self.loss_cal = self.loss_cal.cuda()
        self.gpu = True






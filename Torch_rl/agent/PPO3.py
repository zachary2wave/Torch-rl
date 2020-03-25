import torch
import numpy as np
from Torch_rl.agent.core_policy import Agent_policy_based
from Torch_rl.common.memory import ReplayMemory
from copy import deepcopy
from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable
from gym import spaces
from Torch_rl.common.util import csv_record
from Torch_rl.common.util import gae


class PPO_Agent(Agent_policy_based):
    def __init__(self, env, policy_model, value_model,
                 lr=1e-4, ent_coef=0.01, vf_coef=0.5,
                 ## hyper-parawmeter
                 gamma=0.90, lam=0.95, cliprange=0.2, batch_size=64, value_train_step=10,
                 learning_starts=1000, running_step=2048, running_ep=20, value_regular=0.01,
                 ## decay
                 decay=False, decay_rate=0.9,
                 ##
                 path=None):

        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.cliprange = cliprange

        self.learning_starts = learning_starts
        self.value_train_step = value_train_step

        self.sample_rollout = running_step
        self.sample_ep = running_ep
        self.batch_size = batch_size

        self.loss_cal = torch.nn.SmoothL1Loss()

        self.policy = policy_model
        if value_model == "shared":
            self.value = policy_model
        elif value_model == "copy":
            self.value = deepcopy(policy_model)
        else:
            self.value = value_model

        self.dist = make_pdtype(env.action_space, policy_model)

        policy_model_optim = Adam(self.policy.parameters(), lr=lr)
        value_model_optim = Adam(self.value.parameters(), lr=lr, weight_decay=value_regular)
        if decay:
            self.policy_model_optim = torch.optim.lr_scheduler.ExponentialLR(policy_model_optim, decay_rate,
                                                                            last_epoch=-1)
            self.value_model_optim = torch.optim.lr_scheduler.ExponentialLR(value_model_optim, decay_rate,
                                                                             last_epoch=-1)
        else:
            self.policy_model_optim = policy_model_optim
            self.value_model_optim = value_model_optim

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1, norm_type=2)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1, norm_type=2)

        super(PPO_Agent, self).__init__(path)
        example_input = Variable(torch.rand((100,)+self.env.observation_space.shape))
        self.writer.add_graph(self.policy, input_to_model=example_input)

        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss"]
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss"]

        self.training_round = 0
        self.running_step = 0
        self.record_sample = None
        self.training_step = 0


    def update(self, sample):

        step_len = sample["step_used"]

        time_round = np.ceil(step_len/self.batch_size)
        time_left = time_round*self.batch_size-step_len
        array = list(range(step_len)) +list(range(time_left))

        for train_time in range(time_round):
            index = array[train_time*self.batch_size : (train_time+1)*self.batch_size]
            training_s = self.record_sample["s"][index]
            training_a = self.record_sample["a"][index]
            training_r = self.record_sample["r"][index]
            R = self.record_sample["return"][self.training_step]
            old_value = self.record_sample["value"][self.training_step]
            old_neglogp = self.record_sample["neglogp"][self.training_step]
            advs = self.record_sample["advs"][self.training_step]

            " CALCULATE THE LOSS"
            " Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss"

            #generate Policy gradient loss
            outcome = self.policy.forward(training_s)
            new_policy = self.dist(outcome)
            new_neg_lop = new_policy.neglogp(training_a)
            ratio = torch.exp(old_neglogp - new_neg_lop)
            pg_loss1 = -advs * ratio
            pg_loss2 = -advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = .5 * torch.max(pg_loss1, pg_loss2)

            value_now = self.value.forward(training_s)
            # value loss
            value_clip = old_value + torch.clamp(old_value - value_now, min=-self.cliprange, max=self.cliprange) # Clipped value
            vf_loss1 = self.loss_cal(value_now, R)  # Unclipped loss
            vf_loss2 = self.loss_cal(value_clip, R) # clipped loss
            vf_loss = .5 * torch.max(vf_loss1, vf_loss2)
            # vf_loss = vf_loss1
            # entropy
            entropy = new_policy.entropy().mean()
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            for _ in range(self.value_train_step):
                self.value_model_optim.zero_grad()
                loss.backward(retain_graph=True)
                self.value_model_optim.step()

            self.policy_model_optim.zero_grad()
            loss.backward()
            self.policy_model_optim.step()

            # approxkl = self.loss_cal(neg_log_pac, self.record_sample["neglogp"])
            # self.cliprange = torch.gt(torch.abs(ratio - 1.0).mean(), self.cliprange)
            self.training_step += 1
            if self.training_step == self.record_sample["s"].size()[0]:
                self.training_round += 1
                self.training_step = 0
        return loss.data.numpy(), {"pg_loss": pg_loss.data.numpy(),
                                   "entropy": entropy.data.numpy(),
                                   "vf_loss": vf_loss.data.numpy()}


    def load_weights(self, filepath):
        model = torch.load(filepath+"ppo.pkl")
        self.graph_model.load_state_dict(model["graph_model"])
        self.graph_model_optim.load_state_dict(model["graph_model_optim"])


    def save_weights(self, filepath, overwrite=False):
        torch.save({"graph_model": self.graph_model,
                    "graph_model_optim": self.graph_model_optim,
                    }, filepath + "PPO.pkl")

import torch
import numpy as np
from Torch_rl.agent.core import Agent
from Torch_rl.common.memory import ReplayMemory
from copy import deepcopy
from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable
from gym import spaces
from Torch_rl.common.util import csv_record
from Torch_rl.common.util import gae

class graph_model(torch.nn.Module):
    def __init__(self, policy, value):
        super(graph_model, self).__init__()
        self.policy = policy
        self.value = value

    def forward(self, obs):
        output = self.policy(obs)
        Q = self.value(obs)
        return output, Q

class PPO_Agent(Agent):
    def __init__(self, env, policy_model, value_model,
                 lr=1e-3, ent_coef=0.01, vf_coef=0.5,
                 ## hyper-parawmeter
                 gamma=0.90, lam=0.95, cliprange=0.2,
                 buffer_size=50000, learning_starts=1000, running_step=2000, batch_training_round=20,
                 value_regular=0.01,
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
        self.batch_training_round = batch_training_round
        self.learning_starts = learning_starts
        if running_step =="synchronization":
            self.run_step = 1
        else:
            self.run_step = running_step

        self.replay_buffer = ReplayMemory(buffer_size, ["value", "neglogp"])
        self.loss_cal = torch.nn.SmoothL1Loss()

        self.policy_model = policy_model
        if value_model == "shared":
            self.value_model = policy_model
        elif value_model == "copy":
            self.value_model = deepcopy(policy_model)
        else:
            self.value_model = value_model

        self.graph_model = graph_model(self.policy_model, self.value_model)

        self.run_graph_model = deepcopy(self.graph_model)

        self.dist = make_pdtype(env.action_space, policy_model)

        graph_model_optim = Adam(self.graph_model.parameters(), lr=lr, weight_decay=0.01)
        if decay:
            self.graph_model_optim = torch.optim.lr_scheduler.ExponentialLR(graph_model_optim, decay_rate,
                                                                             last_epoch=-1)
        else:
            self.graph_model_optim = graph_model_optim

        torch.nn.utils.clip_grad_norm_(self.graph_model.parameters(), 1, norm_type=2)

        super(PPO_Agent, self).__init__(path)
        example_input = Variable(torch.rand(100, self.env.observation_space.shape[0]))
        self.writer.add_graph(self.graph_model, input_to_model=example_input)
        self.forward_step_show_list = []
        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss"]
        self.forward_ep_show_list = []
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss"]

        self.training_round = 0
        self.running_step = 0
        self.record_sample = None
        self.training_step = 0

    def forward(self, observation):
        observation = observation.astype(np.float32)
        observation = torch.from_numpy(observation)
        outcome, Q = self.run_graph_model.forward(observation)
        self.pd = self.dist(outcome)
        self.action = self.pd.sample()
        self.Q = Q.data.numpy()
        return self.action.detach().numpy(), self.Q, {}

    def backward(self, sample_):
        sample_["neglogp"] = self.pd.neglogp(self.action)
        sample_["value"] = self. Q
        self.replay_buffer.push(sample_)
        self.running_step += 1
        """"""""""""""
        "training part"
        """"""""""""""
        "1 training start flag"
        "2 have enough sample "
        "3 the training have finished"
        if self.step > self.learning_starts and\
           self.running_step % self.run_step == 0 and\
           self.training_round == 0 and self.training_step == 0:
            " sample advantage generate "
            sample = self.replay_buffer.recent_step_sample(self.running_step)
            last_value = self.value_model.forward(sample["s_"][-1]).unsqueeze(1)
            self.record_sample = gae(sample, last_value, self.gamma, self.lam)
            self.running_step = 0

        "1 need the sample "
        "2 training start flag"
        "3 training round count"
        "4 training at the end of each ep to get the information"
        if self.record_sample is not None and \
           self.step > self.learning_starts and \
           self.training_round < self.batch_training_round:
            "training in single step"
            training_s = self.record_sample["s"][self.training_step]
            training_a = self.record_sample["a"][self.training_step]
            training_r = self.record_sample["r"][self.training_step]
            R = self.record_sample["return"][self.training_step]
            old_value = self.record_sample["value"][self.training_step]
            old_neglogp = self.record_sample["neglogp"][self.training_step]
            advs = self.record_sample["advs"][self.training_step]

            " CALCULATE THE LOSS"
            " Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss"


            #generate Policy gradient loss
            outcome, value_now = self.graph_model.forward(training_s)
            new_policy = self.dist(outcome)
            new_neg_lop = new_policy.neglogp(training_a)
            ratio = torch.exp(old_neglogp - new_neg_lop)
            pg_loss1 = -advs * ratio
            pg_loss2 = -advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = .5 * torch.max(pg_loss1, pg_loss2)
            # value loss
            value_clip = old_value + torch.clamp(old_value - value_now, min=-self.cliprange, max=self.cliprange) # Clipped value
            vf_loss1 = self.loss_cal(value_now, R)  # Unclipped loss
            vf_loss2 = self.loss_cal(value_clip, R) # clipped loss
            vf_loss = .5 * torch.max(vf_loss1, vf_loss2)
            # vf_loss = vf_loss1
            # entropy
            entropy = new_policy.entropy().mean()
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            self.graph_model_optim.zero_grad()
            loss.backward(retain_graph=True)
            self.graph_model_optim.step()

            # approxkl = self.loss_cal(neg_log_pac, self.record_sample["neglogp"])
            # self.cliprange = torch.gt(torch.abs(ratio - 1.0).mean(), self.cliprange)
            self.training_step += 1
            if self.training_step == self.record_sample["s"].size()[0]:
                self.training_round += 1
                self.training_step = 0
            return loss.data.numpy(), {"pg_loss": pg_loss.data.numpy(),
                                       "entropy": entropy.data.numpy(),
                                       "vf_loss": vf_loss.data.numpy()}
        if self.training_round == self.batch_training_round:
            print("this round have training finished")
            self.run_graph_model.load_state_dict(self.graph_model.state_dict())
            self.training_round = 0

        return 0, {"pg_loss": 0, "entropy": 0, "vf_loss": 0}

    def load_weights(self, filepath):
        model = torch.load(filepath+"ppo.pkl")
        self.graph_model.load_state_dict(model["graph_model"])
        self.graph_model_optim.load_state_dict(model["graph_model_optim"])


    def save_weights(self, filepath, overwrite=False):
        torch.save({"graph_model": self.graph_model,
                    "graph_model_optim": self.graph_model_optim,
                    }, filepath + "PPO.pkl")
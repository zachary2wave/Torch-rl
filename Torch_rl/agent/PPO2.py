import torch
import numpy as np
from Torch_rl.agent.core import Agent
from Torch_rl.common.memory import ReplayMemory
from copy import deepcopy
from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable
import random
from Torch_rl.common.util import csv_record
from Torch_rl.common.util import generate_reture,gae

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
                 lr=1e-4, ent_coef=0.01, vf_coef=0.5,
                 ## hyper-parawmeter
                 gamma=0.99, lam=0.95, cliprange=0.2, batch_size = 32,
                 buffer_size=50000, learning_starts=1000, running_step="synchronization", batch_training_round=10,
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
        self.batch_size = batch_size
        self.batch_training_round = batch_training_round
        self.learning_starts = learning_starts
        if running_step =="synchronization":
            self.run_step = 1
        else:
            self.run_step = running_step


        self.replay_buffer = ReplayMemory(buffer_size, ["value", "logp"])
        self.loss_cal = torch.nn.MSELoss()

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

        graph_model_optim = Adam(self.graph_model.parameters(), lr=lr, weight_decay=value_regular)
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

    def forward(self, observation):
        observation = observation.astype(np.float32)
        observation = torch.from_numpy(observation)
        outcome, Q = self.run_graph_model.forward(observation)
        self.pd = self.dist(outcome)
        self.action = self.pd.sample()
        self.Q = Q.data.numpy()
        return self.action.detach().numpy(), self.Q, {}

    def backward(self, sample_):
        sample_["logp"] = self.pd.log_prob(self.action)
        sample_["value"] = self.Q
        self.replay_buffer.push(sample_)
        self.running_step += 1
        """"""""""""""
        "training part"
        """"""""""""""
        if self.step > self.learning_starts and\
           self.running_step % self.run_step == 0 and\
           self.training_round == 0 and sample_["tr"] == 1:
            " sample advantage generate "
            print("")
            with torch.no_grad():
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
           self.training_round < self.batch_training_round and sample_["tr"] == 1:

            start = (self.batch_size * self.batch_training_round) % self.record_sample["s"].size()[0]
            if start+self.batch_size >= self.record_sample["s"].size()[0]:
                end = self.record_sample["s"].size()[0]
            else:
                end = start+self.batch_size
            index = np.arange(start, end)
            " CALCULATE THE LOSS"
            " Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss"
            S = self.record_sample["s"][index].detach()
            A = self.record_sample["a"][index].detach()
            old_log = self.record_sample["logp"][index].detach()
            advs = self.record_sample["advs"][index]
            value = self.record_sample["value"][index].detach()
            returns = self.record_sample["return"][index].detach()
            #generate Policy gradient loss
            outcome, value_now = self.graph_model.forward(S)
            new_policy = self.dist(outcome)
            new_lop = new_policy.log_prob(A)
            ratio = torch.exp(new_lop-old_log)
            pg_loss1 = advs * ratio
            pg_loss2 = advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = -.5 * torch.min(pg_loss1, pg_loss2).mean()
            # value loss
            value_clip = value + torch.clamp(value_now - value, min=-self.cliprange, max=self.cliprange) # Clipped value
            vf_loss1 = self.loss_cal(value_now, returns)   # Unclipped loss
            vf_loss2 = self.loss_cal(value_clip, returns)  # clipped loss
            vf_loss = .5 * torch.max(vf_loss1, vf_loss2)
            # vf_loss = 0.5 * vf_loss1
            # entropy
            entropy = new_policy.entropy().mean()
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            self.graph_model_optim.zero_grad()
            loss.backward(retain_graph=True)
            self.graph_model_optim.step()

            # approxkl = self.loss_cal(neg_log_pac, self.record_sample["neglogp"])
            # self.cliprange = torch.gt(torch.abs(ratio - 1.0).mean(), self.cliprange)
            self.training_round += 1
            return loss.data.numpy(), {"pg_loss": pg_loss.data.numpy(),
                                       "entropy": entropy.data.numpy(),
                                       "vf_loss": vf_loss.data.numpy()}
        if self.training_round == self.batch_training_round:
            print("the"+str(self.episode)+" round have training finished")
            self.run_graph_model.load_state_dict(self.graph_model.state_dict())
            self.training_round = 0
            self.record_sample = None

        return 0, {"pg_loss": 0, "entropy": 0, "vf_loss": 0}

    def load_weights(self, filepath):
        model = torch.load(filepath+"ppo.pkl")
        self.graph_model.load_state_dict(model["graph_model"])
        self.graph_model_optim.load_state_dict(model["graph_model_optim"])


    def save_weights(self, filepath, overwrite=False):
        torch.save({"graph_model": self.graph_model,
                    "graph_model_optim": self.graph_model_optim,
                    }, filepath + "PPO.pkl")
import torch
import numpy as np
from Torch_rl.agent.core_value import Agent_value_based
from Torch_rl.common.memory import ReplayMemory
from copy import deepcopy
from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable
import random
from Torch_rl.common.util import csv_record
from Torch_rl.common.util import generate_reture,gae


class PPO_Agent(Agent_value_based):
    def __init__(self, env, policy_model, value_model,
                 lr=1e-4, ent_coef=0.01, vf_coef=0.5,
                 ## hyper-parawmeter
                 gamma=0.99, lam=0.95, cliprange=0.2, batch_size = 32,
                 buffer_size=50000, learning_starts=1000, running_step="synchronization", batch_training_round=10,
                 value_regular=0.01, train_value_round = 1,
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
        self.train_value_round = train_value_round
        if running_step =="synchronization":
            self.run_step = 1
        else:
            self.run_step = running_step


        self.replay_buffer = ReplayMemory(buffer_size)
        self.loss_cal = torch.nn.MSELoss()

        self.policy_model = policy_model
        if value_model == "shared":
            self.value_model = policy_model
        elif value_model == "copy":
            self.value_model = deepcopy(policy_model)
        else:
            self.value_model = value_model

        self.run_policy_model,self.run_value_model = deepcopy(self.policy_model), deepcopy(self.value_model)

        self.dist = make_pdtype(env.action_space, policy_model)

        policy_model_optim = Adam(self.policy_model.parameters(), lr=lr)
        value_model_optim = Adam(self.value_model.parameters(), lr=lr, weight_decay=value_regular)
        if decay:
            self.policy_model_optim = torch.optim.lr_scheduler.ExponentialLR(policy_model_optim, decay_rate,
                                                                            last_epoch=-1)
            self.value_model_optim = torch.optim.lr_scheduler.ExponentialLR(value_model_optim, decay_rate,
                                                                             last_epoch=-1)
        else:
            self.policy_model_optim = policy_model_optim
            self.value_model_optim = value_model_optim

        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1, norm_type=2)
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1, norm_type=2)

        super(PPO_Agent, self).__init__(path)
        example_input = Variable(torch.rand(100, self.env.observation_space.shape[0]))
        self.writer.add_graph(self.policy_model, input_to_model=example_input)
        self.forward_step_show_list = []
        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss"]
        self.forward_ep_show_list = []
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss"]

        self.training_round = 0
        self.running_step = 0
        self.record_sample = None
        self.loss_record = {"pg_loss": [], "entropy": [], "vf_loss": [], "loss": []}

    def forward(self, observation):
        observation = observation[np.newaxis, :].astype(np.float32)
        observation = torch.from_numpy(observation)
        outcome = self.policy_model.forward(observation)
        self.pd = self.dist(outcome)
        self.action = self.pd.sample()
        self.Q = self.value_model.forward(observation).squeeze()
        return self.action.squeeze(0).detach().numpy(), self.Q.squeeze(0).detach().numpy(), {}

    def backward(self, sample_):
        self.replay_buffer.push(sample_)
        self.running_step += 1
        """"""""""""""
        "training part"
        """"""""""""""
        if self.step > self.learning_starts and self.learning:
            if self.record_sample is None and self.running_step > self.run_step:
                print("***************************************")
                print("In the ", self.episode, "ep")
                sample = self.replay_buffer.recent_step_sample(self.running_step)
                " sample advantage generate "
                sample["value"] = self.value_model.forward(sample["s"]).squeeze()
                last_value = self.value_model.forward(sample["s_"][-1])
                self.record_sample = gae(sample, last_value, self.gamma, self.lam)
                " sample log_probabilty generate"
                outcome = self.policy_model.forward(sample["s"])
                self.pd = self.dist(outcome)
                sample["logp"] = self.pd.log_prob(sample["a"])
                self.loss_record = {"pg_loss": [], "entropy": [], "vf_loss": [], "loss": []}
                self.running_step = 0
            if self.record_sample is not None:
                print("the learning has start...........")
                while self.training_round < self.batch_training_round:
                    start = (self.batch_size * self.training_round) % self.record_sample["s"].size()[0]
                    if start+self.batch_size >= self.record_sample["s"].size()[0]:
                        end = self.record_sample["s"].size()[0]
                    else:
                        end = start+self.batch_size
                    index = np.arange(start, end)
                    S = self.record_sample["s"][index]
                    A = self.record_sample["a"][index]
                    old_log = self.record_sample["logp"][index].detach()
                    advs = self.record_sample["advs"][index].detach()
                    value = self.record_sample["value"][index].detach()
                    returns = self.record_sample["return"][index].detach()

                    " traning the value model"

                    value_now = self.value_model.forward(S)
                    value_clip = value + torch.clamp(value_now - value, min=-self.cliprange, max=self.cliprange) # Clipped value
                    vf_loss1 = self.loss_cal(value_now, returns)   # Unclipped loss
                    vf_loss2 = self.loss_cal(value_clip, returns)  # clipped loss
                    vf_loss = .5 * torch.max(vf_loss1, vf_loss2)  # value loss
                    vf_loss = 0.5 * vf_loss1
                    " CALCULATE THE LOSS"
                    " Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss"

                    #generate Policy gradient loss
                    outcome = self.policy_model.forward(S)
                    new_policy = self.dist(outcome)
                    new_lop = new_policy.log_prob(A)
                    ratio = torch.exp(new_lop-old_log)
                    pg_loss1 = advs * ratio
                    pg_loss2 = advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
                    pg_loss = -.5 * torch.min(pg_loss1, pg_loss2).mean()

                    # entropy
                    entropy = new_policy.entropy().mean()
                    loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

                    self.value_model_optim.zero_grad()
                    loss.backward(retain_graph=True)
                    self.value_model_optim.step()

                    self.policy_model_optim.zero_grad()
                    loss.backward()
                    self.policy_model_optim.step()


                    # approxkl = self.loss_cal(neg_log_pac, self.record_sample["neglogp"])
                    # self.cliprange = torch.gt(torch.abs(ratio - 1.0).mean(), self.cliprange)
                    self.training_round += 1
                    print("round:", self.training_round,
                          "pg_loss:", pg_loss.data.numpy(), "entropy:", entropy.data.numpy(), "vf_loss", vf_loss.data.numpy())
                    self.loss_record["pg_loss"].append(pg_loss.data.numpy())
                    self.loss_record["entropy"].append(entropy.data.numpy())
                    self.loss_record["vf_loss"].append(vf_loss.data.numpy())
                    self.loss_record["loss"].append(loss.data.numpy())
                self.training_round = 0
                self.record_sample = None

        if self.loss_record["loss"] and self.running_step<self.batch_training_round:
            return self.loss_record["loss"][self.running_step],\
                   {"pg_loss": self.loss_record["pg_loss"][self.running_step],
                    "entropy": self.loss_record["vf_loss"][self.running_step],
                    "vf_loss": self.loss_record["loss"][self.running_step]}
        else:
            return 0, {"pg_loss": 0, "entropy": 0, "vf_loss": 0}

    def load_weights(self, filepath):
        model = torch.load(filepath+"ppo.pkl")
        self.policy_model.load_state_dict(model["graph_model"].state_dict())
        self.policy_model_optim.load_state_dict(model["graph_model_optim"])
        self.value_model.load_state_dict(model["value_model"].state_dict())
        self.value_model_optim.load_state_dict(model["value_model_optim"])

    def save_weights(self, filepath, overwrite=False):
        torch.save({"policy_model": self.policy_model,"value_model": self.value_model,
                    "policy_model_optim": self.policy_model_optim,"value_model_optim": self.value_model_optim,
                    }, filepath + "PPO.pkl")
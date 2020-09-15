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
                 gamma=0.99, lam=0.95, cliprange=0.2,
                 buffer_size=50000, learning_starts=1000, running_step=2048, batch_training_round=10,
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

        self.learning_starts = learning_starts
        self.batch_training_round = batch_training_round
        self.run_step = running_step
        self.sample_training_step = self.batch_training_round * self.run_step

        self.replay_buffer = ReplayMemory(buffer_size, ["value", "logp"])
        self.loss_cal = torch.nn.MSELoss()

        self.dist = make_pdtype(env.action_space, policy_model)

        self.policy_model = policy_model
        if value_model == "shared":
            self.value_model = policy_model
        elif value_model == "copy":
            self.value_model = deepcopy(policy_model)
        else:
            self.value_model = value_model

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

        self.run_policy = deepcopy(self.policy_model)
        self.run_value = deepcopy(self.value_model)

        super(PPO_Agent, self).__init__(path)
        example_input = Variable(torch.rand(100, self.env.observation_space.shape[0]))
        self.writer.add_graph(self.policy_model, input_to_model=example_input)
        self.forward_step_show_list = []
        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss"]
        self.forward_ep_show_list = []
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss"]

        self.training_round = 0
        self.training_step = 0
        self.running_step = 0
        self.record_sample = None
        self.train_ticks = np.tile(np.arange(self.run_step), self.batch_training_round)

    def forward(self, observation):
        observation = observation[np.newaxis,:].astype(np.float32)
        observation = torch.from_numpy(observation)
        with torch.no_grad():
            outcome = self.run_policy.forward(observation)
            self.pd = self.dist(outcome)
            self.action = self.pd.sample()
            self.Q = self.run_value.forward(observation)
        return self.action.squeeze(0).detach().numpy(), self.Q.squeeze(0).data.numpy(), {}

    def backward(self, sample_):
        sample_["logp"] = self.pd.log_prob(self.action)
        sample_["value"] = self.Q
        self.replay_buffer.push(sample_)
        self.running_step += 1
        """"""""""""""
        "training part"
        "in each step, we train for batch batch_training_times"
        """"""""""""""
        if self.step > self.learning_starts:
            if self.running_step % self.run_step == 0 and self.training_step == 0:
                " sample advantage generate "
                with torch.no_grad():
                    sample = self.replay_buffer.recent_step_sample(self.running_step)
                    last_value = self.value_model.forward(sample["s_"][-1])
                    self.record_sample = gae(sample, last_value, self.gamma, self.lam)
                self.running_step = 0

            if self.training_step < self.sample_training_step and self.record_sample is not None:
                pg_loss_re = 0
                entropy_re = 0
                vf_loss_re = 0
                loss_re = 0
                for _ in range(self.batch_training_round):
                    index = self.train_ticks[self.training_step]
                    S = self.record_sample["s"][index].detach()
                    A = self.record_sample["a"][index].detach()
                    old_log = self.record_sample["logp"][index].detach()
                    advs = self.record_sample["advs"][index].detach()
                    value = self.record_sample["value"][index].detach()
                    returns = self.record_sample["return"][index].detach()
                    # generate Policy gradient loss
                    outcome = self.run_policy.forward(S)
                    new_policy = self.dist(outcome)
                    new_lop = new_policy.log_prob(A)
                    ratio = torch.exp(new_lop - old_log)
                    pg_loss1 = advs * ratio
                    pg_loss2 = advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
                    pg_loss = -.5 * torch.min(pg_loss1, pg_loss2).mean()
                    # value loss
                    value_now = self.run_value.forward(S)
                    value_clip = value + torch.clamp(value_now - value, min=-self.cliprange,
                                                     max=self.cliprange)  # Clipped value
                    vf_loss1 = self.loss_cal(value_now, returns)  # Unclipped loss
                    vf_loss2 = self.loss_cal(value_clip, returns)  # clipped loss
                    vf_loss = .5 * torch.max(vf_loss1, vf_loss2)
                    # vf_loss = 0.5 * vf_loss1
                    # entropy
                    entropy = new_policy.entropy().mean()
                    loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef
                    # approxkl = self.loss_cal(neg_log_pac, self.record_sample["neglogp"])
                    # self.cliprange = torch.gt(torch.abs(ratio - 1.0).mean(), self.cliprange)

                    self.value_model_optim.zero_grad()
                    loss.backward(retain_graph=True)
                    self.value_model_optim.step()

                    self.policy_model_optim.zero_grad()
                    loss.backward()
                    self.policy_model_optim.step()

                    self.training_step += 1
                    pg_loss_re += pg_loss.data.numpy()
                    entropy_re += entropy.data.numpy()
                    vf_loss_re += vf_loss.data.numpy()
                    loss_re += loss.data.numpy()

                if self.training_step == self.sample_training_step:
                    print("the" + str(self.episode) + " round have training finished")
                    self.run_policy.load_state_dict(self.policy_model.state_dict())
                    self.run_value.load_state_dict(self.value_model.state_dict())
                    self.training_step = 0
                    self.record_sample = None
                return loss_re, {"pg_loss": pg_loss_re, "entropy": entropy_re, "vf_loss": vf_loss_re}
        return 0, {"pg_loss": 0, "entropy": 0, "vf_loss": 0}

    def load_weights(self, filepath):
        model = torch.load(filepath+"ppo.pkl")
        self.policy_model.load_state_dict(model["policy_model"].state_dict())
        self.value_model.load_state_dict(model["value_model"].state_dict())

    def save_weights(self, filepath, overwrite=False):
        torch.save({"policy_model": self.policy_model,"value_model": self.value_model}, filepath + "PPO.pkl")
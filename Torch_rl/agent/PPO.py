import torch
import numpy as np
from Torch_rl.agent.core import Agent
from Torch_rl.common.memory import ReplayMemory
from copy import deepcopy
# from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable
from gym import spaces
from Torch_rl.common.util import csv_record

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

        self.dist = self.make_pdtype(env.action_space)

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

    def forward(self, observation):
        observation = observation.astype(np.float32)
        observation = torch.from_numpy(observation)
        outcome, Q = self.run_graph_model.forward(observation)
        if isinstance(self.env.action_space, spaces.Box):
            # mu = torch.index_select(outcome, -1, torch.arange(0, self.env.action_space.shape[0]))
            # std = torch.index_select(outcome, -1, torch.arange(self.env.action_space.shape[0], self.env.action_space.shape[0]*2))
            self.pd = self.dist(outcome, 1)
            self.action = self.pd.sample()
        else:
            self.pd = self.dist(outcome)
            self.action = self.pd.sample()
        self.Q = Q.data.numpy()
        return self.action.data.numpy(), self.Q, {}

    def backward(self, sample_):
        sample_["neglogp"] = - self.pd.log_prob(self.action)
        sample_["value"] = self. Q
        self.replay_buffer.push(sample_)
        self.running_step += 1
        """"""""""""""
        "training part"
        """"""""""""""
        if self.step > self.learning_starts and\
           self.running_step % self.run_step == 0 and\
           self.training_round == 0 :
            " sample advantage generate "
            sample = self.replay_buffer.recent_step_sample(self.running_step)
            sample["advs"] = torch.zeros((self.running_step, 1), dtype=torch.float32)
            last_value = self.value_model.forward(sample["s_"][-1]).unsqueeze(1)
            lastgaelam = 0
            value = torch.cat((sample["value"], last_value), dim=0)
            for t in reversed(range(self.running_step)):
                delta = sample["r"][t] + self.gamma * value[t+1] * (1-sample["tr"][t]) - value[t]
                lastgaelam = delta + self.gamma * self.lam * (1-sample["tr"][t]) * lastgaelam
                sample["advs"][t] = lastgaelam
            sample["value"] = sample["advs"]+sample["value"]

            # adv = sample["advs"]   # Normalize the advantages
            # adv = (adv - torch.mean(adv))/(torch.std(adv)+1e-8)
            # sample["advs"] = adv
            self.record_sample = sample
            mean_ep_reward = torch.sum(sample["r"])/torch.sum(torch.eq(sample["tr"],1))
            print("the runner have sampled "+str(self.running_step)+" data and the mean_ep_reward is ", mean_ep_reward)
            self.running_step = 0

        "1 need the sample "
        "2 training start flag"
        "3 training round count"
        "4 training at the end of each ep to get the information"
        if self.record_sample is not None and \
           self.step > self.learning_starts and \
           self.training_round < self.batch_training_round and\
           sample_["tr"]==1:
            " CALCULATE THE LOSS"
            " Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss"
            outcome, value_now = self.graph_model.forward(self.record_sample["s"])
            if isinstance(self.env.action_space, spaces.Box):
                # mu = torch.index_select(outcome, -1, torch.arange(0, self.env.action_space.shape[0]))
                # std = torch.index_select(outcome, -1,
                #                          torch.arange(self.env.action_space.shape[0], self.env.action_space.shape[0] * 2))
                self.pd = self.dist(outcome, 1)
            else:
                self.pd = self.dist(outcome)
            csv_record(self.pd.mean.detach().numpy(), "./")
            neg_log_pac = - self.pd.log_prob(self.record_sample["a"])
            entropy = self.pd.entropy().mean()  # Entropy is used to improve exploration by limiting the premature convergence to suboptimal graph.

            value_clip = self.record_sample["value"] + torch.clamp(self.record_sample["value"] - value_now, min=-self.cliprange, max = self.cliprange)

            vf_loss1 = self.loss_cal(value_now, self.record_sample["r"])  # Unclipped loss
            # Clipped value
            vf_loss2 = self.loss_cal(value_clip, self.record_sample["r"])
            vf_loss = .5 * torch.max(vf_loss1, vf_loss2)
            vf_loss = vf_loss1

            ratio = torch.exp(self.record_sample["neglogp"]-neg_log_pac)
            # adv = self.record_sample["advs"]
            pg_loss1 = -self.record_sample["advs"] * ratio
            pg_loss2 = -self.record_sample["advs"] * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = .5 * torch.max(pg_loss1, pg_loss2).mean()

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
            print("this round have training finished")
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

    def make_pdtype(self, ac_space):
        if isinstance(ac_space, spaces.Box):
            from torch.distributions import Normal
            return Normal
        elif isinstance(ac_space, spaces.Discrete):
            from torch.distributions import Categorical
            return Categorical
        elif isinstance(ac_space, spaces.MultiDiscrete):
            from torch.distributions import Categorical
            return Categorical
        elif isinstance(ac_space, spaces.MultiBinary):
            from torch.distributions import Bernoulli
            return Bernoulli
        else:
            raise NotImplementedError


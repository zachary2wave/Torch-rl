import torch
import numpy as np
from Torch_rl.agent.core import Agent
from Torch_rl.common.memory import ReplayMemory
from copy import deepcopy
# from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable
from gym import spaces

class PPO_Agent(Agent):
    def __init__(self, env, policy_model, value_model,
                 lr=1e-5, ent_coef=0.01, vf_coef=0.5,
                 ## hyper-parawmeter
                 gamma=0.90, lam=0.95, cliprange=0.2,
                 buffer_size=50000, learning_starts=1000, running_step=2048, batch_training_round=10,
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
        self.run_step = running_step
        self.batch_training_round = batch_training_round
        self.learning_starts = learning_starts

        self.replay_buffer = ReplayMemory(buffer_size, ["value", "neglogp"])
        self.loss_cal = torch.nn.MSELoss()

        self.policy_model = policy_model
        if value_model == "shared":
            self.value_model = policy_model
        elif value_model == "copy":
            self.value_model = deepcopy(policy_model)
        else:
            self.value_model = value_model

        self.run_policy = deepcopy(self.policy_model)
        self.run_value = deepcopy(self.value_model)

        self.dist = self.make_pdtype(env.action_space)

        policy_model_optim = Adam(self.policy_model.parameters(), lr=lr)
        value_model_optim = Adam(self.value_model.parameters(), lr=lr)
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

        super(PPO_Agent,self).__init__(path)
        example_input = Variable(torch.rand(100, self.env.observation_space.shape[0]))
        self.writer.add_graph(self.policy_model, input_to_model=example_input)
        self.forward_step_show_list = []
        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss"]
        self.forward_ep_show_list = []
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss"]

        self.training_round = 0

    def forward(self, observation):
        observation = observation.astype(np.float32)
        observation = torch.from_numpy(observation)
        outcome = self.run_policy.forward(observation)
        if isinstance(self.env.action_space, spaces.Box):
            mu = torch.index_select(outcome, -1, torch.arange(0, self.env.action_space.shape[0]))
            std = torch.index_select(outcome, -1, torch.arange(self.env.action_space.shape[0], self.env.action_space.shape[0]*2))
            self.pd = self.dist(mu, torch.abs(std))
            self.action = self.pd.sample()
        else:
            self.pd = self.dist(outcome)
            self.action = self.pd.sample()
        Q = self.run_value.forward(observation)
        self.Q = Q.data.numpy()
        return self.action.data.numpy(), self.Q, {}

    def backward(self, sample_):
        sample_["neglogp"] = - self.pd.log_prob(self.action)
        sample_["value"] = self. Q
        self.replay_buffer.push(sample_)
        """"""""""""""
        "training part"
        """"""""""""""
        if self.step % self.run_step == 0 and self.training_round == 0:
            " sample advantage generate "
            sample = self.replay_buffer.recent_step_sample(self.run_step)
            sample["advs"] = np.zeros((self.run_step, 1), dtype=np.float32)
            lastgaelam = 0
            for t in reversed(range(self.run_step)):
                delta = sample["r"][t-1] + self.gamma * sample["value"][t] * (1-sample["tr"][t]) - sample["value"][t-1]
                lastgaelam = delta + self.gamma * self.lam * (1-sample["tr"][t]) * lastgaelam
                sample["advs"][t] = lastgaelam
            sample["value"] = sample["advs"]+sample["value"].numpy()

            adv = sample["advs"]   # Normalize the advantages
            adv = (adv - np.mean(adv))/(np.std(adv)+1e-8)
            sample["advs"] = adv
            self.record_sample = sample

        " training at the end of each ep to get the information"
        if self.step > self.learning_starts and \
           self.training_round < self.batch_training_round and\
           sample_["tr"] == 1:
            " CALCULATE THE LOSS"
            " Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss"
            neg_log_pac = - self.pd.log_prob(self.record_sample["a"])
            entropy = self.pd.entropy().mean()  # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.

            value_now = self.value_model.forward(self.record_sample["s"])

            value_clip = torch.from_numpy(self.record_sample["value"]) + torch.clamp(torch.from_numpy(self.record_sample["value"]) - value_now, min=-self.cliprange, max = self.cliprange)

            vf_loss1 = self.loss_cal(value_now, self.record_sample["r"])    # Unclipped loss
            # Clipped value
            vf_loss2 = self.loss_cal(value_clip, self.record_sample["r"])
            vf_loss = .5 * torch.max(vf_loss1, vf_loss2).mean()

            ratio = torch.exp(self.record_sample["neglogp"]-neg_log_pac)
            adv = torch.from_numpy(adv)
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = .5 * torch.max(pg_loss1, pg_loss2).mean()

            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            self.policy_model_optim.zero_grad()
            loss.backward(retain_graph = True)
            self.policy_model_optim.step()

            self.value_model_optim.zero_grad()
            loss.backward()
            self.value_model_optim.step()

            # approxkl = self.loss_cal(neg_log_pac, self.record_sample["neglogp"])
            self.cliprange = torch.sum(torch.gt(torch.abs(ratio - 1.0), self.cliprange))
            self.training_round += 1
            return loss.data.numpy(), {"pg_loss": pg_loss, "entropy": entropy, "vf_loss": vf_loss}
        if self.training_round == self.batch_training_round:
            self.update_the_running()
            self.training_round = 0
        return 0, {"pg_loss": 0, "entropy": 0, "vf_loss": 0}

    def load_weights(self, filepath):
        model = torch.load(filepath+"ppo.pkl")
        self.policy_model.load_state_dict(model["policy_model"])
        self.value_model.load_state_dict(model["value_model"])
        self.policy_model_optim.load_state_dict(model["policy_model_optim"])
        self.value_model_optim.load_state_dict(model["value_model_optim"])


    def save_weights(self, filepath, overwrite=False):
        torch.save({"actor": self.policy_model, "critic":self.value_model,
                    "policy_model_optim": self.policy_model_optim, "value_model_optim": self.value_model_optim,
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

    def update_the_running(self):
        self.run_value.load_state_dict(self.value_model.state_dict())
        self.run_policy.load_state_dict(self.policy_model.state_dict())
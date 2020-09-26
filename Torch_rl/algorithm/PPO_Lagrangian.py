
from Torch_rl.agent.core_policy import Agent_policy_based
import torch.nn as nn
from copy import deepcopy
from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable
from Torch_rl.common.util import get_gae

class PPO_LAGRANGIAN_Agent(Agent_policy_based):
    def __init__(self, env, policy_model, value_model,
                 lr=5e-4, ent_coef=0.01, vf_coef=0.5,
                 ## hyper-parawmeter
                 gamma=0.99, lam=0.95, cliprange=0.2, batch_size=64, value_train_round=10,
                 running_step=2048, running_ep=20, value_regular=0.01,
                 ## decay
                 decay=False, decay_rate=0.9, lstm_enable=False,
                 ##
                 path=None):
        self.gpu = False
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.cliprange = cliprange

        self.value_train_step = value_train_round

        self.sample_rollout = running_step
        self.sample_ep = running_ep
        self.batch_size = batch_size
        self.lstm_enable = lstm_enable

        self.loss_cal = torch.nn.SmoothL1Loss()

        self.policy = policy_model
        if value_model == "shared":
            self.value = policy_model
        elif value_model == "copy":
            self.value = deepcopy(policy_model)
        else:
            self.value = value_model

        self.cost_value = deepcopy(self.value)

        self.dist = make_pdtype(env.action_space, policy_model)

        self.policy_model_optim = Adam(self.policy.parameters(), lr=lr)
        self.value_model_optim = Adam(self.value.parameters(), lr=lr, weight_decay=value_regular)
        self.cost_value_model_optim = Adam(self.cost_value.parameters(), lr=lr, weight_decay=value_regular)
        if decay:
            self.policy_model_decay_optim = torch.optim.lr_scheduler.ExponentialLR(self.policy_model_optim, decay_rate,
                                                                            last_epoch=-1)
            self.value_model_decay_optim = torch.optim.lr_scheduler.ExponentialLR(self.value_model_optim, decay_rate,
                                                                             last_epoch=-1)
            self.cost_value_model_decay_optim = torch.optim.lr_scheduler.ExponentialLR(self.value_model_optim, decay_rate,
                                                                             last_epoch=-1)

        #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1, norm_type=2)
        #torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1, norm_type=2)

        super(PPO_LAGRANGIAN_Agent, self).__init__(path)
        example_input = Variable(torch.rand((100,)+self.env.observation_space.shape))
        self.writer.add_graph(self.policy, input_to_model=example_input)

        self.backward_step_show_list = ["pg_loss", "entropy", "vf_loss","cost_value"]
        self.backward_ep_show_list = ["pg_loss", "entropy", "vf_loss","cost_value"]

        self.training_round = 0
        self.running_step = 0
        self.record_sample = None
        self.training_step = 0

        self.ui = torch.tensor(1,require_grad=True)
        self.ui_optim = Adam(self.ui, lr=lr)


    def update(self, sample):

        returns, advants = get_gae(sample["r"], sample["tr"], sample["value"], self.gamma,
                                   self.lam)
        sample["advs"] = advants.unsqueeze(1)
        sample["return"] = returns.unsqueeze(1)

        sample["cost"] = []
        for info in sample["info"]:
            sample["cost"].append(info["cost"])

        sample["cost_value"] = self.cost_value.forward(sample["s"])

        returns, advants = get_gae(sample["cost"], sample["tr"], sample["cost_value"], self.gamma,
                                   self.lam)
        sample["cost_advs"] = advants.unsqueeze(1)
        sample["cost_return"] = returns.unsqueeze(1)


        step_len = len(sample["s"])
        if self.lstm_enable:
            flagin = [time for time in range(step_len) if sample["tr"][time]==1]
            time_round = len(flagin)
            array_index = []
            for train_time in range(int(time_round)-1):
                array_index.append(range(flagin[train_time], flagin[train_time+1]))
        else:
            time_round = np.ceil(step_len/self.batch_size)
            time_left = time_round*self.batch_size-step_len
            array = list(range(step_len)) +list(range(int(time_left)))
            array_index = []
            for train_time in range(int(time_round)):
                array_index.append(array[train_time * self.batch_size: (train_time + 1) * self.batch_size])
        loss_re, pgloss_re, enloss_re, vfloss_re = [], [], [], []

        for key in sample.keys():
            temp = torch.stack(list(sample[key]), 0)
            if self.gpu:
                sample[key] = temp.cuda()
            else:
                sample[key] = temp
        for train_time in range(int(time_round)):
            index = array_index[train_time]
        # for index in range(step_len):
            training_s = sample["s"][index].detach()
            training_a = sample["a"][index].detach()
            training_r = sample["r"][index].detach()
            R = sample["return"][index].detach()
            old_value = sample["value"][index].detach()
            old_neglogp = sample["logp"][index].detach()
            advs = sample["advs"][index].detach()
            c_advs = sample["cost_advs"][index].detach()
            c_value = sample["cost_value"][index].detach()
            cost = sample["cost"][index].detach()

            " CALCULATE THE LOSS"
            " Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss"

            " the value loss"
            value_now = self.value.forward(training_s)
            # value loss
            value_clip = old_value + torch.clamp(old_value - value_now, min=-self.cliprange,
                                                 max=self.cliprange)  # Clipped value
            vf_loss1 = self.loss_cal(value_now, R)  # Unclipped loss
            vf_loss2 = self.loss_cal(value_clip, R)  # clipped loss
            vf_loss = .5 * torch.max(vf_loss1, vf_loss2)

            # generate Policy gradient loss
            outcome = self.policy.forward(training_s)
            new_policy = self.dist(outcome)
            new_neg_lop = new_policy.log_prob(training_a)
            ratio = torch.exp(new_neg_lop - old_neglogp)
            pg_loss1 = -advs * ratio
            pg_loss2 = -advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = .5 * torch.max(pg_loss1, pg_loss2).mean()

            # generate Policy gradient loss
            c_pg_loss1 = -c_advs * ratio
            c_pg_loss2 = -c_advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            c_pg_loss = .5 * torch.max(c_pg_loss1, c_pg_loss2).mean()



            # entropy
            entropy = new_policy.entropy().mean()
            loss = pg_loss- self.ui * c_pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

            self.policy_model_optim.zero_grad()
            pg_loss.backward()
            self.policy_model_optim.step()
            for _ in range(self.value_train_step):
                value_now = self.value.forward(training_s)
                # value loss
                value_clip = old_value + torch.clamp(old_value - value_now, min=-self.cliprange,
                                                     max=self.cliprange)  # Clipped value
                vf_loss1 = self.loss_cal(value_now, R)  # Unclipped loss
                vf_loss2 = self.loss_cal(value_clip, R)  # clipped loss
                vf_loss = .5 * torch.max(vf_loss1, vf_loss2)
                self.value_model_optim.zero_grad()
                vf_loss1.backward()
                self.value_model_optim.step()

                cost_now = self.cost_value.forward(training_s)
                cost_vloss = self.loss_cal(cost_now, cost)

                self.cost_value_model_optim.zero_grad()
                cost_vloss.backward()
                self.cost_value_model_optim.step()


            loss_re = loss.cpu().detach().numpy()
            pgloss_re.append(pg_loss.cpu().detach().numpy())
            enloss_re.append(entropy.cpu().detach().numpy())
            vfloss_re.append(vf_loss1.cpu().detach().numpy())
        "training the weights ui"
        for i in sample["cost"]:
            cost = self.ui*sample["cost"]
            self.ui_optim.zero_grad()
            cost.backward()
            self.ui_optim.step()


        return np.sum(loss_re), {"pg_loss": np.sum(pgloss_re),
                                   "entropy": np.sum(enloss_re),
                                   "vf_loss": np.sum(vfloss_re)}

    def load_weights(self, filepath):
        model = torch.load(filepath+"/PPO.pkl")
        self.policy.load_state_dict(model["policy"].state_dict())
        self.value.load_state_dict(model["value"].state_dict())


    def save_weights(self, filepath, overwrite=False):
        torch.save({"policy": self.policy,"value": self.value}, filepath + "/PPO.pkl")


    def cuda(self):
        self.policy.to_gpu()
        self.value.to_gpu()
        self.loss_cal = self.loss_cal.cuda()
        self.gpu = True
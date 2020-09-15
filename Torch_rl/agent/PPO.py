
from Torch_rl.agent.core_policy import Agent_policy_based
import torch.nn as nn
from copy import deepcopy
from Torch_rl.common.distribution import *
from torch.optim import Adam
from torch.autograd import Variable

class PPO_Agent(Agent_policy_based):
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

        self.dist = make_pdtype(env.action_space, policy_model)

        self.policy_model_optim = Adam(self.policy.parameters(), lr=lr)
        self.value_model_optim = Adam(self.value.parameters(), lr=lr, weight_decay=value_regular)
        if decay:
            self.policy_model_decay_optim = torch.optim.lr_scheduler.ExponentialLR(self.policy_model_optim, decay_rate,
                                                                            last_epoch=-1)
            self.value_model_decay_optim = torch.optim.lr_scheduler.ExponentialLR(self.value_model_optim, decay_rate,
                                                                             last_epoch=-1)

        #torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1, norm_type=2)
        #torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1, norm_type=2)

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

            #generate Policy gradient loss
            outcome = self.policy.forward(training_s)
            new_policy = self.dist(outcome)
            new_neg_lop = new_policy.log_prob(training_a)
            ratio = torch.exp(new_neg_lop - old_neglogp)
            pg_loss1 = -advs * ratio
            pg_loss2 = -advs * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            pg_loss = .5 * torch.max(pg_loss1, pg_loss2).mean()

            # entropy
            entropy = new_policy.entropy().mean()
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

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
            # approxkl = self.loss_cal(neg_log_pac, self.record_sample["neglogp"])
            # self.cliprange = torch.gt(torch.abs(ratio - 1.0).mean(), self.cliprange)
            loss_re = loss.cpu().detach().numpy()
            pgloss_re.append(pg_loss.cpu().detach().numpy())
            enloss_re.append(entropy.cpu().detach().numpy())
            vfloss_re.append(vf_loss1.cpu().detach().numpy())
        return np.sum(loss_re), {"pg_loss": np.sum(pgloss_re),
                                   "entropy": np.sum(enloss_re),
                                   "vf_loss": np.sum(vfloss_re)}

    def load_weights(self, filepath):
        model = torch.load(filepath+"/PPO.pkl")
        self.policy.load_state_dict(model["policy"].state_dict())
        self.value.load_state_dict(model["value"].state_dict())


    def save_weights(self, filepath, overwrite=False):
        torch.save({"policy": self.policy,"value": self.value}, filepath + "/PPO.pkl")

    def policy_behavior_clone(self, sample_):
        action_label = sample_["a"].squeeze()
        if self.gpu:
            action_predict = self.policy(sample_["s"].cuda())
            action_label = action_label.cuda()
        else:
            action_predict = self.policy(sample_["s"])
        loss_bc = self.loss_cal(action_label, action_predict)
        del action_label
        del action_predict
        loss = loss_bc
        self.policy_model_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1, norm_type=2)
        self.policy_model_optim.step()
        return loss.cpu().detach().numpy()

    def value_pretrain(self, record_sample, new_sample_len):
        train_times = int(np.floor(new_sample_len/128))
        round_loss = 0
        for io in range(train_times-1):
            index = list(range(128 * io, 128 * (io + 1)))
            if self.gpu:
                predict = torch.from_numpy(np.array(record_sample["s"])[index]).cuda()
                lable = torch.from_numpy(np.array(record_sample["return"]))[index].cuda()
            else:
                predict = torch.from_numpy(np.array(record_sample["s"])[index])
                lable = torch.from_numpy(np.array(record_sample["return"]))[index]
            value_now = self.value.forward(predict)
            # value loss
            vf_loss = self.loss_cal(value_now, lable)  # Unclipped loss
            del predict
            del lable
            self.value_model_optim.zero_grad()
            vf_loss.backward()
            self.value_model_optim.step()
            round_loss += vf_loss.cpu().detach().numpy()
        return round_loss

    def cuda(self):
        self.policy.to_gpu()
        self.value.to_gpu()
        self.loss_cal = self.loss_cal.cuda()
        self.gpu = True
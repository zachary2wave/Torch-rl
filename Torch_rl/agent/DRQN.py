import torch
import numpy as np
from Torch_rl.common.memory import ReplayMemory_Sequence
from Torch_rl.agent.core_value import Agent_value_based
from copy import deepcopy
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from Torch_rl.common.loss import huber_loss
from torch.autograd import Variable

class Dueling_dqn(nn.Module):
    def __init__(self, model, dueling_way):
        super(Dueling_dqn, self).__init__()
        self.dueling_way = dueling_way
        self.model_layer = model.linears[:-1]
        layer_infor = model.layer_infor
        self.A_est = nn.Linear(layer_infor[-2], layer_infor[-1])
        self.V_est = nn.Linear(layer_infor[-2], 1)

    def forward(self, obs):
        x = obs
        for layer in self.model_layer:
            x = layer(x)
        A = F.relu(self.A_est(x))
        V = self.V_est(x)
        if self.dueling_way == "native":
            A = A
        elif self.dueling_way == "mean":
            A = A - torch.max(A)
        elif self.dueling_way == "avg":
            A = A - torch.mean(A)
        return V - A


class gpu_foward(nn.Module):
    def __init__(self, model):
        super(gpu_foward, self).__init__()
        model.to_gpu()
        self.model = model
    def forward(self,obs):
        obs = obs.cuda()
        out = self.model(obs)
        return out


class DRQN_Agent(Agent_value_based):
    def __init__(self, env, model, policy,
                 ## hyper-parameter
                 gamma=0.90, lr=1e-3,  learning_starts=1000,
                 target_network_update_freq=500,
                 ## memory
                 batch_size=32, buffer_size=50000, max_seq_len=2000, replay_len=100,
                 ## decay
                 decay=False, decay_rate=0.9,
                 ## DDqn && DuelingDQN
                 double_dqn=True, dueling_dqn=False, dueling_way="native",
                 ## prioritized_replay
                 prioritized_replay=False,
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, param_noise=False,
                 ##
                 path=None):

        """

        :param env:      the GYM environment
        :param model:    the Torch NN model
        :param policy:   the policy when choosing action
        :param ep:       the MAX episode time
        :param step:     the MAx step time
         .........................hyper-parameter..................................
        :param gamma:
        :param lr:
        :param batchsize:
        :param buffer_size:
        :param target_network_update_freq:
        .........................further improve way..................................
        :param double_dqn:  whether enable DDQN
        :param dueling_dqn: whether dueling DDQN
        :param dueling_way: the Dueling DQN method
            it can choose the following three ways
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
        .........................prioritized-part..................................
        :param prioritized_replay: (bool) if True prioritized replay buffer will be used.
        :param prioritized_replay_alpha: (float)alpha parameter for prioritized replay buffer.
        It determines how much prioritization is used, with alpha=0 corresponding to the uniform case.
        :param prioritized_replay_beta0: (float) initial value of beta for prioritized replay buffer
        :param prioritized_replay_beta_iters: (int) number of iterations over which beta will be annealed from initial
            value to 1.0. If set to None equals to max_timesteps.
        :param prioritized_replay_eps: (float) epsilon to add to the TD errors when updating priorities.
        .........................imitation_learning_part..................................
        :param imitation_learning_policy:     To initial the network with the given policy
        which is supervised way to training the network
        :param IL_time:    supervised training times
        :param network_kwargs:
        """
        self.gpu = False
        self.env = env
        self.policy = policy

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq
        self.double_dqn = double_dqn
        if dueling_dqn:
            self.Q_net = Dueling_dqn(model, dueling_way)
        else:
            self.Q_net = model

        self.target_Q_net = deepcopy(self.Q_net)

        q_net_optim = Adam(self.Q_net.parameters(), lr=lr)
        if decay:
            self.optim = torch.optim.lr_scheduler.ExponentialLR(q_net_optim, decay_rate, last_epoch=-1)
        else:
            self.optim = q_net_optim

        self.replay_buffer = ReplayMemory_Sequence(buffer_size, max_seq_len, other_record=["h","c"])

        self.replay_buffer.batch_size = batch_size
        self.replay_buffer.sequence_len = replay_len
        if replay_len < max_seq_len:
            self.replay_sample = self.replay_buffer.sample_sequence
        else:
            self.replay_sample = self.replay_buffer.sample_ep
        self.learning = False
        super(DRQN_Agent, self).__init__(path)
        example_input = Variable(torch.rand((replay_len, 100)+self.env.observation_space.shape))
        self.writer.add_graph(self.Q_net, input_to_model=example_input)
        self.forward_step_show_list = []
        self.backward_step_show_list =[]
        self.forward_ep_show_list = []
        self.backward_ep_show_list = []

        self.h_state = model.init_H_C(1)

    def forward(self, observation):
        observation = observation[np.newaxis, np.newaxis, :].astype(np.float32)
        observation = torch.from_numpy(observation)
        Q_value, self.h_state = self.Q_net.forward(observation, self.h_state)
        Q_value = Q_value.cpu().squeeze().detach().numpy()
        if self.policy is not None:
            action = self.policy.select_action(Q_value)
        else:
            action = np.argmax(Q_value)
        return action, np.max(Q_value), {}

    def backward(self, sample_):
        sample_["h"] = self.h_state[0].detach().numpy()
        sample_["c"] = self.h_state[1].detach().numpy()
        self.replay_buffer.push(sample_)
        if self.step > self.learning_starts and self.learning:
            sample = self.replay_sample()
            if self.gpu:
                for key in sample.keys():
                    sample[key] = sample[key].cuda()
            assert sample["s"].size(1) == self.batch_size
            a = sample["a"].long()
            Q, H = self.Q_net(sample["s"])
            Q  = Q.gather(2, a)
            if self.double_dqn:
                Q_next, H = self.Q_net(sample["s_"])
                _, next_actions = Q_next.max(2, keepdim=True)
                Qtarget_next, H = self.Q_net(sample["s_"])
                targetQ = Qtarget_next.gather(2, next_actions)
            else:
                Qtarget_next, H = self.target_Q_net(sample["s_"])
                targetQ, next_actions = Qtarget_next.max(2, keepdim=True)

            expected_q_values = sample["r"] + self.gamma * targetQ * (1.0 - sample["tr"])
            loss = torch.mean(huber_loss(expected_q_values-Q))
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), 1, norm_type=2)
            self.optim.step()
            if self.step % self.target_network_update_freq == 0:
                self.target_net_update()
            loss = loss.data.numpy()
            return loss, {}
        return 0, {}

    def target_net_update(self):
        self.target_Q_net.load_state_dict(self.Q_net.state_dict())

    def load_weights(self, filepath):
        model = torch.load(filepath+'DQN.pkl')
        self.Q_net.load_state_dict(model["Q_net"].state_dict())
        self.target_Q_net.load_state_dict(model["target_Q_net"].state_dict())
        # self.optim.load_state_dict(model["optim"])

    def save_weights(self, filepath, overwrite=True):
        torch.save({"Q_net": self.Q_net,
                    "target_Q_net": self.target_Q_net,
                    "optim": self.optim
                    }, filepath+"DQN.pkl")

    def cuda(self):
        self.Q_net = gpu_foward(self.Q_net)
        self.target_Q_net = deepcopy(self.Q_net)
        self.gpu = True












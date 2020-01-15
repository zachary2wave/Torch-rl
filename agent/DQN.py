import torch
import numpy as np
from common.memory import ReplayMemory
from agent.core import Agent
from copy import deepcopy
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
from common.loss import huber_loss
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

class DQN_Agent(Agent):
    def __init__(self, env, model, policy,
                 ## hyper-parameter
                 gamma=0.90, lr=1e-3, batch_size=32, buffer_size=50000, learning_starts=1000,
                 target_network_update_freq=500,
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

        self.replay_buffer = ReplayMemory(buffer_size)
        self.learning = False
        super(DQN_Agent, self).__init__(path)
        example_input = Variable(torch.rand(100, self.env.observation_space.shape[0]))
        self.writer.add_graph(self.Q_net, input_to_model=example_input)

    def forward(self, observation):
        observation = observation.astype(np.float32)
        observation = torch.from_numpy(observation)
        Q_value = self.Q_net.forward(observation)
        Q_value = Q_value.detach().numpy()
        if self.policy is not None:
            action = self.policy.select_action(Q_value)
        else:
            action = np.argmax(Q_value)
        return action, Q_value

    def backward(self, sample_):
        self.replay_buffer.push(sample_)
        if self.step > self.learning_starts and self.learning:
            sample = self.replay_buffer.sample(self.batch_size)
            assert len(sample["s"]) == self.batch_size
            a = sample["a"].long().unsqueeze(1)
            Q = self.Q_net(sample["s"]).gather(1, a)
            if self.double_dqn:
                _, next_actions = self.Q_net(sample["s_"]).max(1, keepdim=True)
                targetQ = self.target_Q_net(sample["s_"]).gather(1, next_actions)
            else:
                targetQ = self.target_Q_net(sample["s_"]).max(1, keepdim=True)
            targetQ = targetQ.squeeze(1)
            Q = Q.squeeze(1)
            expected_q_values = sample["r"] + self.gamma * targetQ * (1.0 - sample["tr"])
            loss = torch.mean(huber_loss(expected_q_values-Q))
            self.Q_net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), 1, norm_type=2)
            self.optim.step()
            if self.step % self.target_network_update_freq == 0:
                self.target_net_update()
            loss = loss.data.numpy()
            return loss
        return 0

    def target_net_update(self):
        self.target_Q_net.load_state_dict(self.Q_net.state_dict())

    def load_weights(self, filepath):
        model = torch.load(filepath)
        self.Q_net.load_state_dict(model["Q_net"])
        self.target_Q_net.load_state_dict(model["target_Q_net"])
        self.optim.load_state_dict(model["optim"])

    def save_weights(self, filepath, overwrite=True):
        torch.save({"Q_net": self.Q_net,
                    "target_Q_net": self.target_Q_net,
                    "optim": self.optim
                    }, filepath+"DQN.pkl")
















import torch
import numpy as np
from Torch_rl.common.memory import ReplayMemory
from Torch_rl.agent.core_value import Agent_value_based
from copy import deepcopy
from torch.optim import Adam
from torch import nn
from Torch_rl.common.loss import huber_loss
from torch.autograd import Variable

class actor_critic(nn.Module):
    def __init__(self, actor, critic):
        super(actor_critic, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, obs):
        a = self.actor(obs)
        input = torch.cat((obs, a), axis=-1)
        Q = self.critic(input)
        return Q


class gpu_foward(nn.Module):
    def __init__(self, model):
        super(gpu_foward, self).__init__()
        model.to_gpu()
        self.model = model

    def forward(self, obs):
        obs = obs.cuda()
        out = self.model(obs)
        return out



class DDPG_Agent(Agent_value_based):
    def __init__(self, env, actor_model, critic_model,
                 actor_lr=1e-4, critic_lr=1e-3,
                 actor_target_network_update_freq=1000, critic_target_network_update_freq=1000,
                 actor_training_freq=1, critic_training_freq=1,
                 ## hyper-parameter
                 gamma=0.99, batch_size=32, buffer_size=50000, learning_starts=1000,
                 ## lr_decay
                 decay=False, decay_rate=0.9, critic_l2_reg=1e-2, clip_norm =None,
                 ##
                 path=None):

        self.gpu = False
        self.env = env

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.replay_buffer = ReplayMemory(buffer_size)

        self.actor_training_freq, self.critic_training_freq = actor_training_freq, critic_training_freq
        self.actor_target_network_update_freq = actor_target_network_update_freq
        self.critic_target_network_update_freq = critic_target_network_update_freq
        self.actor = actor_model
        self.critic = critic_model
        self.target_actor = deepcopy(actor_model)
        self.target_critic = deepcopy(critic_model)

        self.actor_critic = actor_critic(self.actor, self.critic)

        actor_optim = Adam(self.actor.parameters(), lr=actor_lr)
        critic_optim = Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_l2_reg)
        if decay:
            self.actor_optim = torch.optim.lr_scheduler.ExponentialLR(actor_optim, decay_rate, last_epoch=-1)
            self.critic_optim = torch.optim.lr_scheduler.ExponentialLR(critic_optim, decay_rate, last_epoch=-1)
        else:
            self.actor_optim = actor_optim
            self.critic_optim = critic_optim

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1, norm_type=2)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1, norm_type=2)
        super(DDPG_Agent, self).__init__(path)
        example_input = Variable(torch.rand(100, self.env.observation_space.shape[0]))
        self.writer.add_graph(self.actor_critic, input_to_model=example_input)
        self.forward_step_show_list = []
        self.backward_step_show_list =[]
        self.forward_ep_show_list = []
        self.backward_ep_show_list = []

    def forward(self, observation):
        observation = observation[np.newaxis, :].astype(np.float32)
        observation = torch.from_numpy(observation)
        action = self.actor.forward(observation)
        action = torch.normal(action, torch.ones_like(action))
        Q = self.critic(torch.cat((observation, action), dim=1)).squeeze().detach().numpy()
        action = action.data.numpy()
        return action, Q, {}

    def backward(self, sample_):
        self.replay_buffer.push(sample_)
        if self.step > self.learning_starts and self.learning:
            sample = self.replay_buffer.sample(self.batch_size)
            if self.gpu:
                for key in sample.keys():
                    sample[key] = sample[key].cuda()
            assert len(sample["s"]) == self.batch_size
            "update the critic "
            if self.step % self.critic_training_freq == 0:
                input = torch.cat((sample["s"], sample["a"]), -1)
                Q = self.critic(input)
                target_a = self.target_actor(sample["s_"])
                target_input = torch.cat((sample["s_"], target_a), -1)
                targetQ = self.target_critic(target_input)
                targetQ = targetQ.squeeze(1)
                Q = Q.squeeze(1)
                expected_q_values = sample["r"] + self.gamma * targetQ * (1.0 - sample["tr"])
                loss = torch.mean(huber_loss(expected_q_values - Q))
                self.critic_optim.zero_grad()
                loss.backward()
                self.critic_optim.step()
            "training the actor"
            if self.step % self.actor_training_freq == 0:
                # Q = self.critic(torch.cat((sample["s"], self.actor(sample["s"])), -1))
                Q = self.actor_critic.forward(sample["s"])
                Q = -torch.mean(Q)
                self.actor_optim.zero_grad()
                Q.backward()
                self.actor_optim.step()
            if self.step % self.actor_target_network_update_freq == 0:
                self.target_actor_net_update()
            if self.step % self.critic_target_network_update_freq == 0:
                self.target_critic_net_update()
            loss = loss.data.numpy()
            return loss, {}
        return 0, {}

    def target_actor_net_update(self):
        self.target_actor.load_state_dict(self.actor.state_dict())

    def target_critic_net_update(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def load_weights(self, filepath):
        model = torch.load(filepath)
        self.actor.load_state_dict(model["actor"])
        self.critic.load_state_dict(model["critic"])
        self.target_actor.load_state_dict(model["target_actor"])
        self.target_critic.load_state_dict(model["target_critic"])
        self.actor_optim.load_state_dict(model["actor_optim"])
        self.critic_optim.load_state_dict(model["critic_optim"])


    def save_weights(self, filepath, overwrite=False):
        torch.save({"actor": self.actor, "critic":self.critic,
                    "target_actor": self.target_actor,"target_critic": self.target_critic,
                    "actor_optim": self.actor_optim, "critic_optim": self.critic_optim
                    }, filepath + "DDPG.pkl")

    def cuda(self):
        self.actor = gpu_foward(self.actor)
        self.critic = gpu_foward(self.critic)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.gpu = True



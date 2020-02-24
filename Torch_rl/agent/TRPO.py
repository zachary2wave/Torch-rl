import torch
from Torch_rl.agent.core import Agent
from Torch_rl.common.memory import ReplayMemory

class TRPO_Agent(Agent):
    def __init__(self, env, actor_model, critic_model,
                 actor_lr=1e-5, critic_lr=1e-5,
                 actor_target_network_update_freq=3000, critic_target_network_update_freq=3000,
                 actor_training_freq=1, critic_training_freq=1,
                 ## hyper-parameter
                 gamma=0.90, batch_size=32, buffer_size=50000, learning_starts=1000,
                 ## decay
                 decay=False, decay_rate=0.9,
                 ##
                 path=None):
        self.env = env

        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts

        self.replay_buffer = ReplayMemory(buffer_size)

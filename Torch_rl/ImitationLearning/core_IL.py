from copy import deepcopy
from abc import ABC
import numpy as np
import torch
from Torch_rl.common import logger
from Torch_rl.common.logger import CSVOutputFormat


class Agent_IL(ABC):

    """
    Abstract base class for all implemented imitation_learning algorithms.

    the class contains the following methods

    we define the episode as the agent finished the training
    and the step as the agent interact with the env once
    """

    def __init__(self, path):
        self.step = 0
        self.episode = 0
        """
        config the logfile 
        """
        configlist = ["stdout", "log", 'tensorboard', "csv"]
        if path is None:
            path = "./"
        logger.configure(path, configlist)
        self.csvwritter = CSVOutputFormat(path + "record_trajectory.csv")
        loggerCEN = logger.get_current().output_formats[configlist.index('tensorboard')]
        self.writer = loggerCEN.writer
        self.path = path

    def training_with_data(self,expert_policy, max_imitation_learning_step=1e5,
                            max_ep_cycle=2000, buffer_size=32, learning_start = 1000):
        raise NotImplementedError()

    def training_with_policy(self):
        raise NotImplementedError()



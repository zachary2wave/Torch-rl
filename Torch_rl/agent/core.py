from copy import deepcopy
from abc import ABC
import numpy as np
from Torch_rl.common import logger
from Torch_rl.common.logger import CSVOutputFormat
import torch


class Agent(ABC):
    """
    所有算法的父类
    Abstract base class for all implemented agents.

    其中包含了
    - `forward`   前向传播、计算action
    - `backward`  后向传播、更新网络
    - `load_weights`  加载权重
    - `save_weights`  存储权重
    - `layers`        网络层
    - 'forward'       前向传播
    - 'backward'       前向传播
    定义 episode 完成一次为一个episode
    定义 step    为交互一次
    """
    def __init__(self, path):
        self.step = 0
        self.episode = 0
        """
        config the logfile 
        """
        configlist = ["stdout", "log", 'tensorboard']
        logger.configure(path, configlist)
        self.csvwritter = CSVOutputFormat(path+"record_trajectory.csv")
        loggerCEN = logger.get_current().output_formats[configlist.index('tensorboard')]
        self.writer = loggerCEN.writer


    def imitation_learning(self):
        pass

    def train(self, max_step=None, max_ep_cycle=2000, verbose=2, render=False, record_ep_inter=None):
        self.learning = True
        print("the train phase ........")
        self.interact(max_step=max_step, max_ep_cycle=max_ep_cycle, render=render,
                 verbose=verbose, record_ep_inter=record_ep_inter)

    def test(self, max_step=None, max_ep_cycle=2000, verbose=2, render=False, record_ep_inter=None):
        self.learning = False
        self.learning_starts = 0
        self.step = 0
        self.episode = 0
        print("the test phase ........")
        self.interact(max_step=max_step, max_ep_cycle=max_ep_cycle, render=render,
                 verbose=verbose, record_ep_inter=record_ep_inter)

    def interact(self, max_step=50000, max_ep_cycle=2000, render = False,
            verbose=1, record_ep_inter=None):
        '''
        :param max_step:
        :param max_ep_time:
        :param max_ep_cycle:  max step in per circle
        .........................show parameter..................................
        :param verbose
        if verbose == 1   show every ep
        if verbose == 2   show every step
        :param record_ep_inter
        record_ep_interact data
        :return: None
        '''
        # if IL_time is not None:

        # .....................initially——recode...........................#
        ep_reward = []
        ep_Q_value = []
        ep_loss = []

        while self.step < max_step:
            s = self.env.reset()
            'reset the ep record'
            ep_r, ep_q, ep_l = 0, 0, 0
            'reset the RL flag'
            ep_cycle, done = 0, 0
            ep_show={}
            if not self.backward_ep_show_list:
                for key in self.backward_ep_show_list:
                    ep_show[key]=0
            self.episode += 1
            while done == 0 and ep_cycle < max_ep_cycle:
                self.step += 1
                ep_cycle += 1
                'the interaction part'
                a, Q, info_forward = self.forward(s)
                # print(a)
                s_, r, done, info = self.env.step(a)
                sample = {"s": s, "a": a, "s_": s_, "r": r, "tr": done}
                s = s_
                loss, info_backward = self.backward(sample)
                if render:
                    self.env.render()
                'the record part'

                if verbose == 1 and self.step > self.learning_starts:
                    logger.record_tabular("steps", self.step)
                    logger.record_tabular("episodes", self.episode)
                    logger.record_tabular("loss", loss)
                    logger.record_tabular("reward", r)
                    logger.record_tabular("Q", Q)
                    if not self.forward_step_show_list:
                        for key in self.forward_step_show_list:
                            logger.record_tabular(key, info_forward[key])
                    if not self.backward_step_show_list:
                        for key in self.backward_step_show_list:
                            logger.record_tabular(key, info_backward[key])
                    logger.dump_tabular()
                if record_ep_inter is not None:
                    if self.episode % record_ep_inter == 0:
                        kvs = {"s": s, "a": a, "s_": s_, "r": r,
                               "tr": done, "ep": self.episode, "step": self.step, "ep_step": ep_cycle}
                        self.csvwritter.writekvs(kvs)
                ep_r += r
                ep_q += Q
                ep_l += loss
                if not self.backward_ep_show_list:
                    for key in self.backward_ep_show_list:
                        ep_show[key] += info_backward[key]
                if done:
                    ep_reward.append(ep_r)
                    ep_Q_value.append(ep_q)
                    ep_loss.append(ep_l)
                    mean_100ep_reward = round(np.mean(ep_reward[-101:-1]), 1)
                    if verbose == 2 and self.step > self.learning_starts:
                        logger.record_tabular("steps", self.step)
                        logger.record_tabular("episodes", self.episode)
                        logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                        logger.record_tabular("episode_reward", ep_reward[-1])
                        logger.record_tabular("episode_reward_per_step", ep_reward[-1]/ep_cycle)
                        logger.record_tabular("episode_loss_per_step", ep_l/ep_cycle)
                        logger.record_tabular("episode_Q_value_per_step", ep_q/ep_cycle)
                        logger.record_tabular("step_used", ep_cycle)
                        if not self.forward_ep_show_list:
                            for key in self.forward_ep_show_list:
                                logger.record_tabular(key, info_forward[key])
                        if not self.backward_ep_show_list:
                            for key in self.backward_ep_show_list:
                                logger.record_tabular(key, ep_show[key])
                        logger.dump_tabular()

    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        raise NotImplementedError()

    def backward(self, sample):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.

        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.

        # Returns
            List of metrics values
        """
        raise NotImplementedError()

    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.

        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        raise NotImplementedError()

    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.

        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        raise NotImplementedError()

    def Imitation_Learning(self, step_time, data=None, policy=None, verbose=2):
        '''
        :param data:  the data is a list, and each element is a dict with 5 keys s,a,r,s_,tr
        sample = {"s": s, "a": a, "s_": s_, "r": r, "tr": done}
        :param policy:
        :return:
        '''
        if data is not None and policy is not None:
            raise Exception("The IL only need one way to guide, Please make sure the input ")

        if data is not None:
            for time in step_time:
                self.step += 1
                loss = self.backward(data[time])
                if verbose == 1:
                    logger.record_tabular("steps", self.step)
                    logger.record_tabular("loss", loss)
                    logger.dumpkvs()

        if policy is not None:
            s = self.env.reset()
            for time in step_time:
                self.step += 1
                a = policy(s)
                s_, r, done, info = self.env.step(a)
                sample = {"s": s, "a": a, "s_": s_, "r": r, "tr": done}
                loss = self.backward(sample)
                s = s_
                if verbose == 1:
                    logger.record_tabular("steps", self.step)
                    logger.record_tabular("loss", loss)
                    logger.dumpkvs()










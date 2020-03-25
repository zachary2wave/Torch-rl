from copy import deepcopy
from abc import ABC
import numpy as np
import torch
from Torch_rl.common import logger
from Torch_rl.common.logger import CSVOutputFormat
from Torch_rl.common.memory import ReplayMemory
from Torch_rl.common.distribution import *
from Torch_rl.common.util import gae



class Agent_policy_based(ABC):
    """
    所有算法的父类
    Abstract base class for all implemented agents.

    其中包含了
    - `runner`    根据policy 产生 sample
    - `learning`  根据Sample 训练网络
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
        configlist = ["stdout", "log", 'tensorboard', "csv"]
        if path is None:
            path = "./"
        logger.configure(path, configlist)
        self.csvwritter = CSVOutputFormat(path+"record_trajectory.csv")
        loggerCEN = logger.get_current().output_formats[configlist.index('tensorboard')]
        self.writer = loggerCEN.writer
        self.path = path


    def imitation_learning(self):
        pass

    def train(self, max_step=None, max_ep_cycle=2000, verbose=2, learning_start=1000, render=False, record_ep_inter=None):
        self.learning = True
        print("the train phase ........")
        self.interact(max_step=max_step, max_ep_cycle=max_ep_cycle, learning_start=learning_start, render=render,
                 verbose=verbose, record_ep_inter=record_ep_inter)

    def test(self, max_step=None, max_ep_cycle=2000, verbose=2, render=False, record_ep_inter=None):
        self.learning = False
        self.learning_starts = 0
        self.step = 0
        self.episode = 0
        print("the test phase ........")
        self.interact(max_step=max_step, max_ep_cycle=max_ep_cycle, render=render,
                 verbose=verbose, record_ep_inter=record_ep_inter)

    def interact(self, max_step=50000, max_ep_cycle=2000, train_rollout=10,learning_start=1000,
                 render = False, verbose=1, record_ep_inter=None):
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
        self.render = render

        # .....................initially——recode...........................#
        rollout = 0
        now_best_reward = -np.inf

        self.dist = make_pdtype(self.env.action_space, self.policy)
        sample_generate = self.runner(self.sample_rollout, self.sample_ep, max_ep_cycle, record_ep_inter)
        while self.step < max_step:
            sample = next(sample_generate)
            record_sample = gae(sample["buffer"], sample["last_Q"], self.gamma, self.lam)
            rollout += 1

            if self.step > learning_start:
                ep_show = {}
                if self.backward_ep_show_list:
                    for key in self.backward_ep_show_list:
                        ep_show[key] = 0
                rollout_loss = 0
                for time in range(train_rollout):
                    loss, other_infor = self.update(record_sample)
                    if verbose == 1:
                        logger.record_tabular("1.train_rollout", time)
                        logger.record_tabular("2.loss", loss)
                        flag = 3
                        if self.backward_step_show_list:
                            for key in self.backward_step_show_list:
                                logger.record_tabular(str(flag) +"."+ key, other_infor[key])
                                flag += 1
                        logger.dump_tabular()
                    rollout_loss += loss
                    if self.backward_ep_show_list:
                        for key in self.backward_ep_show_list:
                            ep_show[key] += other_infor[key]
                if verbose == 2:
                    logger.record_tabular("01.steps", self.step)
                    logger.record_tabular("02.episodes", self.episode)
                    logger.record_tabular("03.rollouts", rollout)
                    logger.record_tabular("04.rollout_loss", rollout_loss)
                    # logger.record_tabular("05.episode_loss_per_step", rollout_loss / samole["step_used"])
                    # logger.record_tabular("06.episode_Q_value", sample["ep_Q_value"])
                    logger.record_tabular("07.episode_Q_value_per_ep", np.mean(sample["ep_Q_value"]))
                    logger.record_tabular("08.mean_ep_step_used", np.mean(sample["ep_step_used"]))
                    flag = 10
                    if self.backward_ep_show_list:
                        for key in self.backward_ep_show_list:
                            logger.record_tabular(str(flag) + "." + key, ep_show[key])
                            flag += 1
                    logger.dump_tabular()
            if np.mean(sample["ep_reward"])>now_best_reward:
                self.save_weights(self.path)
                print("the best mean ep reward is ", np.mean(sample["ep_reward"]), "the weight is saved")
                now_best_reward = np.mean(sample["ep_reward"])

    def runner(self, sample_step=None, sample_ep=None, max_ep_step=2000, record_ep_inter=None):
        if sample_step is not None:
            buffer = ReplayMemory(sample_step, ["value", "logp"])
        else:
            buffer = ReplayMemory(sample_ep*max_ep_step, ["value", "logp"])
        s = self.env.reset()
        ep_reward, ep_Q_value, ep_step_used = [], [], []
        ep_r, ep_q, ep_cycle = 0, 0, 0
        while True:
            s = s[np.newaxis, :].astype(np.float32)
            s = torch.from_numpy(s)
            with torch.no_grad():
                outcome = self.policy.forward(s)
                Q = self.value.forward(s)
            pd = self.dist(outcome)
            a = pd.sample()

            s_, r, done, info = self.env.step(a)
            if self.render:
                self.env.render()
            ep_r += r
            ep_q += Q
            ep_cycle +=1
            self.step += 1

            logp = pd.log_prob(a)
            sample_ = {
                "s": s,
                "a": a,
                "r": torch.tensor([r]),
                "tr": torch.tensor([int(done)]),
                "s_":torch.from_numpy(s_),
                "logp": logp, "value": Q}
            buffer.push(sample_)
            s = deepcopy(s_)

            if record_ep_inter is not None:
                if self.episode % record_ep_inter == 0:
                    kvs = {"s": s, "a": a, "s_": s_, "r": r,
                           "tr": done, "ep": self.episode, "step": self.step, "ep_step": ep_cycle}
                    self.csvwritter.writekvs(kvs)

            if done:
                s = self.env.reset()
                self.episode += 1
                ep_reward.append(ep_r)
                ep_Q_value.append(ep_q)
                ep_step_used.append(ep_cycle)
                ep_r, ep_q, ep_cycle = 0, 0, 0

            if sample_step is not None:
                if self.step > 0 and self.step % sample_step==0:
                    s_ = torch.from_numpy(s_.astype(np.float32))
                    with torch.no_grad():
                        last_Q = self.value.forward(s_)
                    print("now is we have sampled for :", self.step , "and" , self.episode,"\n",
                          "this round have sampled for " + str(sample_step) + " steps, ", len(ep_reward), "episode",
                          "and the mean reward per step is",  np.mean(buffer.memory["r"]),
                          "the mean ep reward is ", np.mean(ep_reward))
                    yield {"buffer": buffer.memory,
                           "ep_reward": ep_reward,
                           "ep_Q_value": ep_Q_value,
                           "ep_step_used": ep_step_used,
                           "ep_used": len(ep_reward),
                           "step_used": sample_step,
                           "last_Q" : last_Q
                           }
                    ep_reward, ep_Q_value = [], []
                    if sample_step is not None:
                        buffer = ReplayMemory(sample_step, ["value", "logp"])
                    else:
                        buffer = ReplayMemory(sample_ep * max_ep_step, ["value", "logp"])

            else:
                if self.step > 0 and self.episode % sample_ep==0:
                    s_ = torch.from_numpy(s_.astype(np.float32))
                    last_Q = self.value.forward(s_)
                    print("now is we have sampled for :", self.step , "and" , self.episode,"\n",
                          "this round have sampled for " + str(sample_step) + " steps, ", len(ep_reward), "episode",
                          "and the mean reward per step is",  np.mean(buffer.memory["r"]),
                          "the mean ep reward is ", np.mean(ep_reward))
                    yield {"buffer": buffer.memory,
                           "ep_reward": ep_reward,
                           "ep_Q_value": ep_Q_value,
                           "ep_step_used": ep_step_used,
                           "ep_used": sample_ep,
                           "step_used": len(buffer.memory["tr"]),
                           "last_Q": last_Q
                           }
                    ep_reward, ep_Q_value = [], []
                    if sample_step is not None:
                        buffer = ReplayMemory(sample_step, ["value", "logp"])
                    else:
                        buffer = ReplayMemory(sample_ep * max_ep_step, ["value", "logp"])


    def update(self, sample):
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

    def cuda(self):
        """
        use the cuda
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










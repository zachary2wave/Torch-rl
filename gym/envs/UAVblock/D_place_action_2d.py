import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from gym.envs.classic_control import rendering
import matplotlib.animation as animation


# import panda as pd
import scipy.io as sio
'''
根据C2D更改了状态和动作
主要修改在134行  
'''

class D_place_action_2d(gym.Env):
    def __init__(self):


        # 定义场景
        self.NUAV = 1 # UAV 及 用户参数
        self.NSP = 7
        self.MAXboundary = 300
        self.place_action = 5
        # 定义 障碍物
        self.blockstartx = [50, -200, -200, 100, -250]
        self.blockstarty = [150, -200, 0, -150, 200]
        self.blockw = [100, 160, 70, 80, 130]
        self.blockl = [140, 80, 170, 130, 40]

        # self.SPplacex = np.array([])
        # self.SPplacey = np.array([])
        # for i in range(self.NSP):
        #     inarea = 1
        #     while inarea:
        #         SPplacex = np.random.randint(-self.MAXboundary , self.MAXboundary , 1)  # 节点位置x
        #         SPplacey = np.random.randint(-self.MAXboundary , self.MAXboundary , 1)  # 节点位置y
        #         inarea = self.inblock(SPplacex, SPplacey)
        #     self.SPplacex = np.append(self.SPplacex, SPplacex)
        #     self.SPplacey = np.append(self.SPplacey, SPplacey)

        self.a_sp = []
        achoose = np.linspace(-1*self.place_action,  self.place_action, num=5)
        for i in range(0, len(achoose)):
            for j in range(0, len(achoose)):
                self.a_sp.append([achoose[i], achoose[j]])

        self.action_space = spaces.Discrete(len(achoose) ** 2)
        self.observation_space = spaces.Box(low=-150, high=150, shape=(4,), dtype=np.float32)
        self.viewer = None

    def reset(self):
        # 初始化强化学习
        self.done = 0
        self.time = 0
        # 初始化无人机 和 node节点 数据
        self.placex = 0  # 无人机位置x
        self.placey = 0  # 无人机位置y
        inarea = 1
        while inarea > 0:
            self.goalx = np.random.randint(-self.MAXboundary, self.MAXboundary, 1)  # 节点位置x
            self.goaly = np.random.randint(-self.MAXboundary, self.MAXboundary, 1)  # 节点位置y
            inarea = self.inblock(self.goalx, self.goaly)

        S = [self.placex / self.MAXboundary, self.placey / self.MAXboundary]
        S = np.append(S, [self.goalx / self.MAXboundary, self.goaly / self.MAXboundary])
        return S

    def step(self, a):
        # print(self.goalx, self.goaly, self.placex, self.placey)
        self.time += 1
        self.a = np.array(self.a_sp[a])
        x = self.placex + self.a[0]
        y = self.placey + self.a[1]
        tp1 = self.bound(x, self.MAXboundary, -self.MAXboundary)
        tp2 = self.bound(y, self.MAXboundary, -self.MAXboundary)
        tp4 = self.inblock(x, y)
        olddis = sum([abs(self.placex - self.goalx) + abs(self.placey - self.goaly)])

        reward = 0
        reward += -(tp1 + tp2 + bool(tp4))
        if tp4==0:
            self.placex = self.placex + (1 - tp1) * self.a[0]  # 无人机位置x
            self.placey = self.placey + (1 - tp2) * self.a[1]  # 无人机位置y

        dis = sum([abs(self.placex-self.goalx)+abs(self.placey-self.goaly)])
        # reward = olddis[0] - dis[0]
        reward += -1
        done = 0
        if dis < 10 or self.time > 2000:
            done = 1
        S = [self.placex/self.MAXboundary, self.placey/self.MAXboundary]
        S_ = np.append(S, [self.goalx/self.MAXboundary, self.goaly/self.MAXboundary])
        return S_, reward, done, {}

    def bound(self, v, upband, lowband):
        flag = int(v > upband) + int(v < lowband)
        return flag

    def inblock(self,x,y):
        blocknum = len(self.blockstartx)
        flag = 0
        for time in range(blocknum):
            if self.blockstartx[time] < x < self.blockstartx[time]+self.blockw[time] \
                    and self.blockstarty[time] < y < self.blockstarty[time]+self.blockl[time]:
                flag = time+1
                break
        return flag

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 800
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            xs = np.linspace(100, 700, 100)
            ys = 100*np.ones_like(xs)
            xys = list(zip(xs, ys))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            xs = np.linspace(100, 700, 100)
            ys = 700*np.ones_like(xs)
            xys = list(zip(xs, ys))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            ys = np.linspace(100, 700, 100)
            xs = 700*np.ones_like(xs)
            xys = list(zip(xs, ys))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            ys = np.linspace(100, 700, 100)
            xs = 100*np.ones_like(xs)
            xys = list(zip(xs, ys))
            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            for i in range(len(self.blockstartx)):
                block = rendering.FilledPolygon([(self.blockstartx[i]+screen_width/2,  self.blockstarty[i]+screen_height/2),
                                           (self.blockstartx[i]+screen_width/2, self.blockstarty[i]+self.blockl[i]+screen_height/2),
                                           (self.blockstartx[i]+self.blockw[i]+screen_width/2, self.blockstarty[i]+self.blockl[i]+screen_height/2),
                                           (self.blockstartx[i]+self.blockw[i]+screen_width/2, self.blockstarty[i]+screen_height/2)])

                self.viewer.add_geom(block)

            aim_mark = rendering.make_circle(3, filled=True)
            aim_mark.set_color(0, 1, 1)
            aim_mark.add_attr(rendering.Transform(translation=(0, 0)))
            self.aim_transform = rendering.Transform()
            aim_mark.add_attr(self.aim_transform)
            self.viewer.add_geom(aim_mark)

            uav_mark = rendering.make_circle(5, filled=True)
            uav_mark.set_color(1, 0, 0)
            uav_mark.add_attr(rendering.Transform(translation=(0, 0)))
            self.uav_transform = rendering.Transform()
            uav_mark.add_attr(self.uav_transform)
            self.viewer.add_geom(uav_mark)
        self.uav_transform.set_translation(self.placex+screen_width/2, self.placey+screen_width/2)
        self.aim_transform.set_translation(self.goalx+screen_width/2, self.goaly+screen_height/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
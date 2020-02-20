import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# import panda as pd
import scipy.io as sio
import time as TT
'''
11.2 建立碰撞模型 

'''
class DL3dwithblock(gym.Env):
    def __init__(self):
        # UAV 及 用户参数
        self.NUAV = 1
        self.NSP = 3
        self.Vmax = 20                                 # 最大速度    in m/s
        self.Vmaxz = 10
        self.amax = 5                                # 最大加速度  in m^2/s
        self.MAXp, self.MAXpzlow, self.MAXpzhigh = 400, 40, 100
        self.delta = 0.5                             # 时隙
        self.T = 1000                                  # 总时间
        self.time = 0

        self.N = self.T/self.delta
        self.Pmax = 10                                 #  dBm  功率
        self.choose = 'Urban'
        self.SNRlimit = 0
        self.K = 0.01                                  # 空气阻力系数
        self.alpha = 4
        # 环境信道参数
        self.done = 0
        self.B = 1e6                                   # 带宽 1Mhz
        self.N0 = -130                                 # dBm
        self.m = 1900                                  # in g
        self.R_th = 1.3*self.B                         #
        self.inta_los, self.inta_Nlos = 0, 0
        if self.choose == 'subruban':
                self.inta_los = 0.1
                self.inta_Nlos = 21
        elif self.choose == 'Urban':
                self.inta_los = 1
                self.inta_Nlos = 20
        elif self.choose == 'DenseUrban':
                self.inta_los = 1.6
                self.inta_Nlos = 23
        elif self.choose == 'HighUrban':
                self.inta_los = 2.3
                self.inta_Nlos = 34

        f = 3e9  # 载频
        c = 3e8  # 光速
        self.lossb = 20*math.log10(f*4*math.pi/c)
        '''
        block 区域 其中a 为 左上点坐标 b为 右下点坐标 
        也就是ax一定小于bx  ay一定大于by
        '''
        self.blockstartx = [50, -100, -200]
        self.blockstarty = [150, -50, 0]
        self.blockw = [100, 80, 90]
        self.blockl = [70, 90, 60]
        self.blockh = [70, 60, 40]
        self.block = []

        for time in range(len(self.blockstarty)):
            point = [
                [self.blockstartx[time],self.blockstarty[time], 0],
                [self.blockstartx[time]+self.blockw[time], self.blockstarty[time], 0],
                [self.blockstartx[time], self.blockstarty[time]+self.blockl[time], 0],
                [self.blockstartx[time]+self.blockw[time], self.blockstarty[time] + self.blockl[time], 0],
                [self.blockstartx[time], self.blockstarty[time], self.blockh[time]],
                [self.blockstartx[time] + self.blockw[time], self.blockstarty[time], self.blockh[time]],
                [self.blockstartx[time], self.blockstarty[time] + self.blockl[time], self.blockh[time]],
                [self.blockstartx[time] + self.blockw[time], self.blockstarty[time] + self.blockl[time], self.blockh[time]]
                     ]
            self.block.append(point)

        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.a = np.array([0, 0, 0])                            # 加速度
        self.v = np.array([10, 10, 0])                          # 速度
        self.place = [0, 0, 50]       # 无人机位置x                                   # 无人机位置y



        # 定义动作空间
        self.a_sp = []
        achoose = np.linspace(-self.amax, self.amax, num=5)
        achoosez = np.linspace(-self.amaxz, self.amaxz, num=5)
        for i in range(0, len(achoose)):
            for j in range(0, len(achoose)):
                for k in range(0,len(achoosez)):
                    self.a_sp.append([achoose[i], achoose[j], achoosez[k]])
        self.action_space = spaces.Discrete(len(achoose) * len(achoose) * len(achoosez))
        "利用了用户的信噪比 和 数据量 + 无人机位置 x,y 和 无人机速度 vx,vy "
        self.observation_space = spaces.Box(low=-150, high=150, shape=(6,), dtype=np.float32)
        # 定义输出的变量
        self.show = 0
        self.boundary = 1

    def reset(self):
        self.a = np.array([0, 0, 0])  # 加速度
        self.v = np.array([10, 10, 0])  # 速度
        self.place = [0, 0, 50]  # 无人机位置
        self.time = 0
        inarea = 1
        while inarea:
            placex= np.random.randint(-self.MAXp , self.MAXp , 1)  # 节点位置x
            placey = np.random.randint(-self.MAXp , self.MAXp , 1)  # 节点位置y
            placez = np.random.randint(self.MAXpzlow, self.MAXpzhigh, 1)  # 节点位置z
            inarea = self.inblock(placex, placey,placez)

        inarea = 1
        while inarea:
            self.goalx = np.random.randint(-self.MAXp , self.MAXp, 1)  # 节点位置x
            self.goaly = np.random.randint(-self.MAXp , self.MAXp, 1)  # 节点位置y
            self.goalz = np.random.randint(self.MAXpzlow, self.MAXpzhigh, 1)
            inarea = self.inblock(self.goalx, self.goaly, self.goaly)
        self.time = 0


        S = np.array([self.place[0] / self.MAXp,
                      self.place[1] / self.MAXp,
                      self.place[2] / (self.MAXpzhigh-self.MAXpzlow)])
        S = np.append(S, [v/self.Vmax for v in self.v])
        return S

    def step(self, a):
        reward = 0
        done = 0
        self.time +=1
        self.a = np.array(self.a_sp[a])
        self.P = self.Pmax
        P = 10**(self.P/10)/1000
        outbound = 0
        for i in range(len(self.v)):
            tempbound = self.bound(a[i] * self.delta + self.v[i], self.Vmax, -self.Vmax)
            self.v[i] += (1 - tempbound) * self.a[i] * self.delta
            outbound += tempbound
        reward += -(outbound) * 10

        x, y, z = [self.place[i] + self.v[i] * self.delta + 0.5 * self.a[i] * (self.delta ** 2) for i in
                   range(len(self.v))]
        tp1 = self.bound(x, self.MAXp, -self.MAXp)
        tp2 = self.bound(y, self.MAXp, -self.MAXp)
        tp3 = self.bound(z, self.MAXpzhigh, self.MAXpzlow)
        tp4 = self.inblock(x, y, z)
        if tp4:
            reward += -(tp1 + tp2 + tp3 + tp4) * 10
        else:
            self.place = [(1 - tp1) * x + tp1 * self.place[0],
                          (1 - tp2) * y + tp2 * self.place[1],
                          (1 - tp3) * x + tp3 * self.place[2]]
            # 判断所链接的用户
            olddis = sum(abs([self.goal[i]-self.place[i] for i in range(len(self.goal))]))
            self.place = [x, y, z]
            # 判断所链接的用户
            dis = sum(abs([self.goal[i]-self.place[i] for i in range(len(self.goal))]))
            # 判断信噪比
            reward += olddis-dis

        if dis < 10:
            done = 1
        S_ = np.array([self.place[0] / self.MAXp,
                      self.place[1] / self.MAXp,
                      self.place[2] / (self.MAXpzhigh-self.MAXpzlow)])
        S_ = np.append(S_, [v/self.Vmax for v in self.v])
        return S_, reward/100, done, {}
# 判断生成的点是否在可行范围内
    def inblock(self, x, y, z):
        blocknum = len(self.blockstartx)
        flag = 0
        for time in range(blocknum):
            if self.blockstartx[time] < x < self.blockstartx[time] + self.blockw[time] \
                    and self.blockstarty[time] < y < self.blockstarty[time] + self.blockl[time] \
                    and z < self.blockh[time]:
                flag = 1
                break
        return flag

        # 判断速度是否在可行范围内

    def speedbound(self, vx, vy, vz):
        flag = int(vx > self.Vmax) + int(vx < -self.Vmax) \
               + int(vy > self.Vmax) + int(vy < -self.Vmax) \
               + int(vz > self.Vmaxz) + int(vy < -self.Vmaxz)
        return flag

        # 判断飞行位置是否在可行范围内

    def placebound(self, x, y, z):
        flag = int(x > self.placemax) + int(x < -self.placemax) + \
               int(y > self.placemax) + int(y < -self.placemax) + \
               int(z < 0) + int(z > self.placemaxz)
        return flag

    # 计算瞬时信噪比
    def Rate(self):
        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        for i in range(self.NSP):
            D = math.sqrt((self.placex - self.SPplacex[i]) ** 2 +
                          (self.placey - self.SPplacey[i]) ** 2 +
                          (self.placez) ** 2)
            K = (self.placey - self.SPplacey[i]) / (self.placex - self.SPplacex[i])
            flag = 0
            for bl in range(len(self.blockax)):
                dot1 = np.sign(K * (self.blockax[bl] - self.placex) - (self.blockay[bl] - self.placey))
                dot2 = np.sign(K * (self.blockax[bl] - self.placex) - (self.blockby[bl] - self.placey))
                dot3 = np.sign(K * (self.blockbx[bl] - self.placex) - (self.blockby[bl] - self.placey))
                dot4 = np.sign(K * (self.blockbx[bl] - self.placex) - (self.blockay[bl] - self.placey))
                if dot1 > 0 and dot2 > 0 and dot3 > 0 and dot4 > 0:
                    flag = 1
                elif dot1 < 0 and dot2 < 0 and dot3 < 0 and dot4 < 0:
                    flag = 1
            PLmax = 10 * math.log10(D ** self.alpha) + self.lossb + flag * self.inta_los + (1 - flag) * self.inta_Nlos
            SNR[i] = self.P-PLmax-self.N0
            rate[i] = self.B*math.log2(1+self.IdB(SNR[i]))
        return rate, SNR

# 计算顺时功率
    def P_calfly(self):
        C1 = 9.26e-4
        C2 = 2250
        g  = 9.8
        normV = np.linalg.norm(self.v)
        norma = np.linalg.norm(self.a)
        cos = np.sum(self.v*self.a)
        Ps = C1*normV**3+C2/normV*(1+(norma**2-cos**2/normV**2)/(g**2))
        return Ps

    def relative_position(self):
        replacex = self.SPplacex - self.placex
        replacey = self.SPplacey - self.placey
        return replacex, replacey
# 计算dB
    def dB(self,a):
        b = 10*math.log10(a/10)
        return b

    def IdB(self, a):
        b = math.pow(10,a/10)
        return b
# 画图三维
    def drawplot(self):
        fig = plt.figure(1)
        ax = Axes3D(fig)
        ax.scatter(self.placex, self.placey, 100)
        ax.scatter(self.SPplacex, self.SPplacey, np.zeros_like(self.SPplacex))
        ax.text(self.placex, self.placey, self.placez,
                'loc='+str([self.placex, self.placey, self.placez])+'\n'
                +'V='+str(self.v)+'\n'+'P='+str(self.P))
        if self.cline != -1:
            ax.plot([self.placex, self.SPplacex[self.cline]], [self.placey, self.SPplacey[self.cline]],
                    [self.placez, 0], '--')
            ax.text((self.placex + self.SPplacex[self.cline])/2, (self.placey+self.SPplacey[self.cline])/2,
                    (self.placez + 0)/2, str(self.rate[self.cline]))
            ax.text(self.SPplacex[self.cline], self.SPplacex[self.cline], self.SPplacex[self.cline],
                    'loc='+str(self.SPplacex[self.cline])+str(self.SPplacex[self.cline])+'\n'
                     +'G='+str(self.G[self.cline])+'\n')
        ax.set_xlim(-self.MAXp, self.MAXp)
        ax.set_ylim(-self.MAXp, self.MAXp)
        ax.set_zlim(0, 150)
        plt.self.show()
    def trajectory(self):
        fig = plt.figure(1)
        ax = Axes3D(fig)
        trax = self.data[:, 0]
        tray = self.data[:, 1]
        ax.set_xlim(-self.MAXp , self.MAXp )
        ax.set_ylim(-self.MAXp , self.MAXp )
        ax.set_zlim(0, 120)
        ax.plot3D(trax, tray, 100*np.ones_like(trax), 'r')
        ax.scatter3D(self.SPplacex, self.SPplacey, np.zeros_like(self.SPplacex), 'g')
        for cline in range(self.NUAV):
            ax.text(self.SPplacex[cline], self.SPplacey[cline], 0,
                'loc=' + str(self.SPplacex[cline]) + str(self.SPplacey[cline]) + '\n'
                + 'G=' + str(self.G[cline]))
        plt.self.show()

    def render(self, mode='human'):
        return {}



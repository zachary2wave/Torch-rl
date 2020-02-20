import gym
from gym import spaces
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from gym.utils import seeding
# import panda as pd
import scipy.io as sio
import time as TT
from gym.envs.classic_control import rendering
'''
本环境是利用下行链路
每一次计算强化学习动作是一个时隙。
这个时隙之内速度，位置不变？

此版本为最简单的版本
支持断点重传
始终是跟最大的相连

本次尝试利用范围内的相对位置作为状态

'''

class Downlink_C2Ddy(gym.Env):
    def __init__(self):
        # UAV 及 用户参数
        self.NUAV = 1
        self.NSP = 5
        self.Vmax = 20                                 # 最大速度    in m/s
        self.Vmaxz = 10
        self.amax = 5                                 # 最大加速度  in m^2/s
        self.delta = 0.5                               # 时隙
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

        f = 3e9  # 载频
        c = 3e8  # 光速
        self.lossb = 20*math.log10(f*4*math.pi/c)
        self.viewer = None
        # 初始参数  说明 向上加速度为正 向下 加速度为负
        self.a = np.array([0, 0])                             # 加速度
        self.v = np.array([10, 10])                          # 速度
        self.placex = 0                                          # 无人机位置x
        self.placey = 0                                          # 无人机位置y
        self.placez = 100                                         # 无人机位置z
        self.SPplacex = np.random.randint(-400, 400, self.NSP)   # 节点位置x
        self.SPplacey = np.random.randint(-400, 400, self.NSP)   # 节点位置y
        self.G0 = np.random.uniform(100, 150, self.NSP)            # 每个节点的数据量 M为单位
        # self.G = self.G0[:]
        self.P = 10                                              # 初始发射功率dBm
        self.P_data = 5                                          # 处理功率  单位W
        self.PLmax = self.PLoss()
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.rate)
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        # 定义状态空间
        # ax = spaces.Box(low=-self.amax, high=self.amax, shape=(1,), dtype=np.float32)
        # ay = spaces.Box(low=-self.amax, high=self.amax, shape=(1,), dtype=np.float32)
        # p = spaces.Box(low=0, high=self.Pmax, shape=(1,), dtype=np.float32)
        # self.action_space = np.array([ax, ay])
        self.action_space = spaces.Box(low=-self.amax, high=self.amax, shape=(2,), dtype=np.float32)
        # v_space = spaces.Box(low=-self.Vmax, high=self.Vmax, shape=(1,), dtype=np.float32)
        # p_space = spaces.Box(low=-200, high=200, shape=(1,), dtype=np.float32)
        # o_space = spaces.Box(low=-200, high=200, shape=(1,), dtype=np.float32)
        #
        # SNR = spaces.Box(low=-50, high=150, shape=(1,), dtype=np.float32)
        # Gleft = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        # task = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32)
        # # 状态格式 前面12个 分为三组 分别是无人机感知范围内最大的3个节点的X 和 Y  和 连接的信噪比 和剩余数据
        # # 然后是无人机自身的位置 x 和 Y vx xy  加速度ax 和 ay  信噪比 功率
        # tempspace = [p_space, p_space, Gleft] * self.NSP
        # tempspace.append(o_space), tempspace.append(o_space)
        # tempspace.append(v_space), tempspace.append(v_space)
        # tempspace.append(task)

        # self.observation_space = np.array(tempspace)  # 无人机
        self.observation_space =spaces.Box(low=-1, high=1, shape=(2*self.NSP+5,), dtype=np.float32)
        self.data = [self.placex, self.placey, self.v[0], self.v[1], self.a[0], self.a[1], self.P, 0, self.cline, 0, 0]
        self.show = 0
        self.boundary = 1

        self.tarx=[]
        self.tary=[]

    def reset(self):
        # 初始化强化学习
        self.done = 0
        # 初始化无人机 和 node节点 数据
        self.a = [0, 0]  # 加速度
        self.v = [10, 10]  # 速度
        self.placex = 0  # 无人机位置x
        self.placey = 0  # 无人机位置y
        self.placez = 100  # 无人机位置z
        self.SPplacex = np.random.randint(-200, 200, self.NSP)  # 节点位置x
        self.SPplacey = np.random.randint(-200, 200, self.NSP)  # 节点位置y
        self.G0 = np.random.uniform(100, 150, self.NSP)
        self.G = np.copy(self.G0)    # 每个节点的数据量

        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.SNR)

        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
        S = []
        # repositionx, repositiony= self.relative_position()
        temp = np.concatenate(([g/150 for g in self.G], [snr/10 for snr in self.SNR]), axis=0)
        S = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S = np.append(S, [self.placex/400, self.placey/400])
        S = np.append(S, [v/self.Vmax for v in self.v])
        S = np.append(S, 0)
        self.totalP = 0
        self.totalR = 0
        self.steptime = 0

        self.tarx = []
        self.tary = []
        return S

    def step(self, a):
        reward = 0
        bound_reward = 0
        self.steptime += 1
        total = 0
        # old_totalP = self.totalP
        # old_totalR = self.totalR
        # old_e = old_totalR/(old_totalP+1)
        # TT.sleep(1)
        acc = a[0:2]
        self.P = self.Pmax
        P = 10**(self.P/10)/1000    #   W 为单位
        # 速度、位置变化
        # self.a = self.a + acc - self.K*self.v
        self.a = acc
        self.v += self.a*self.delta

        self.placex += self.v[0]*self.delta                              # 无人机位置x
        self.placey += self.v[1]*self.delta                                 # 无人机位置y
        if self.show == 1:
            print('action=', a)
            print('speed=', self.v)
        # 判断所链接的用户
        self.rate, self.SNR = self.Rate()
        self.cline = np.argmax(self.rate)
        # 判断信噪比
        PS = self.P_calfly() # 动力消耗功率
        output = 0
        if self.SNR[self.cline] <= self.SNRlimit:
            self.cline = -1
            reward += -0.165
        else:
            Gidea = self.rate[self.cline]*self.delta/1e6
            if Gidea < self.G[self.cline]:
                output = Gidea
                self.G[self.cline] -= output
            else:
                output = self.G[self.cline]
                self.G[self.cline] = 0

        # if output == 0:
        #     reward +=-1
        # elif output>0:
        #     reward += 1
        reward += -1
        # reward += (self.totalR*0.9 + output) / (self.totalP*0.9 + P + PS + self.P_data) * 1000


        self.totalP += P + PS + self.P_data
        self.totalR += output
        # total = self.totalR / self.totalP
        # if output <= 0:
        #     reward += (self.totalR / self.totalP - old_e) * 1e3
        # else:
        #     reward += (self.totalR / self.totalP / 0.9 - old_e) * 1e3
        # reward += output / (P + PS + self.P_data) * 1000

            # reward += self.totalR/self.totalP*1e2
                # 删除
                # self.SPplacex = np.delete(self.SPplacex, self.cline)
                # self.SPplacey = np.delete(self.SPplacey, self.cline)
                # self.G = np.delete(self.G, self.cline)


        teskleft = np.sum(self.G)
        if teskleft == 0 or self.steptime == 1000:
            self.done = 1

            # reward += sum(self.G0)
        # 限制最大用时
        # if self.time == self.T:
        #     self.done = 1
        # 限制约束范围
        if self.placex > 400:
            # reward += -10e-2 * np.maximum(abs(self.placex) - 300, 0)-0.1
            reward +=-10
            bound_reward +=-10
            if self.boundary == 1:
                self.placex = 400
            if self.show == 1:
                print('place out of self.boundary1 ')
        if self.placex < -400:
            # reward += -10e-2 * np.maximum(abs(self.placex) - 300, 0)-0.1
            reward += -10
            bound_reward += -10
            if self.boundary == 1:
                self.placex = -400
            if self.show == 1:
                print('place out of self.boundary 2')

        if self.placey > 400:
            # reward += -10e-2 * np.maximum(abs(self.placey) - 300, 0)-0.1
            reward += -10
            bound_reward += -10
            if self.boundary == 1:
                self.placey = 400
            if self.show == 1:
                print('place out of self.boundary3 ')
        if self.placey < -400:
            # reward += -10e-2 * np.maximum(abs(self.placey) - 300, 0)-0.1
            reward += -10
            bound_reward += -10
            if self.boundary == 1:
                self.placey = -400
            if self.show == 1:
                print('place out of self.boundary4 ')
        # 限制速度范围
        if self.v[0] > self.Vmax:
            # reward += - 50e-2 * np.maximum(abs(self.v[0]) - 40, 0)
            reward += -10
            bound_reward += -10
            if self.boundary == 1:
                self.v[0] = self.Vmax
            if self.show == 1:
                print('speed out of self.boundary1 ')
        if self.v[0] < -self.Vmax:
            # reward += - 50e-2 * np.maximum(abs(self.v[0]) - 40, 0)
            reward += -10
            bound_reward += -10
            if self.boundary == 1:
                self.v[0] = -self.Vmax
            if self.show == 1:
                print('speed out of self.boundary2 ')
        if self.v[1] > self.Vmax:
            # reward += - 50e-2 * np.maximum(abs(self.v[1]) - 40, 0)
            reward += -10
            bound_reward += -10
            if self.boundary == 1:
                self.v[1] = self.Vmax
            if self.show == 1:
                print('speed out of self.boundary3 ')

        if self.v[1] < -self.Vmax:
            # reward += - 50e-2 * np.maximum(abs(self.v[1]) - 40, 0)
            reward += -10
            bound_reward += -10
            if self.boundary == 1:
                self.v[1] = -self.Vmax
            if self.show == 1:
                print('speed out of self.boundary 4')

        # repositionx, repositiony = self.relative_position()
        temp = np.concatenate(([g/150 for g in self.G], [snr/10 for snr in self.SNR]), axis=0)
        S_ = np.reshape(np.transpose(temp), (2 * self.NSP, 1))
        S_ = np.append(S_, [self.placex/400, self.placey/400])
        S_ = np.append(S_, [v/self.Vmax for v in self.v])
        S_ = np.append(S_, self.cline)
        # print(self.placex,self.placey)
        # {'efficent': total, 'cline': self.cline, 'Ux': self.placex, 'Uy': self.placey,
        #  'Spx': self.SPplacex, 'Spy': self.SPplacey, 'G': self.G, "Vx": self.v[0], "Vy": self.v[1]}

        return S_, reward, self.done, {"BD": bound_reward}
# 计算瞬时信噪比
    def Rate(self):
        PLmax, D = self.PLoss()
        rate = np.zeros(self.NSP)
        SNR = np.zeros(self.NSP)
        for i in range(self.NSP):
            SNR[i] = self.P-PLmax[i]-self.N0
            rate[i] = self.B*math.log2(1+self.IdB(SNR[i]))
        return rate, SNR

# 计算瞬时时间延迟 loss
    def PLoss(self):
        caij = np.zeros(shape=[4, 4])
        cbij = np.zeros(shape=[4, 4])
        caij[0, :] = [9.34e-1, 2.30e-1, -2.25e-3, 1.86e-5]
        caij[1, :] = [1.97e-2, 2.44e-3, 6.58e-6, 0]
        caij[2, :] = [-1.24e-4, -3.34e-6, 0, 0]
        caij[3, :] = [2.73e-7, 0, 0, 0]
        cbij[0, :] = [1.17, -7.56e-2, 1.98e-3, -1.78e-5]
        cbij[1, :] = [-5.79e-3, 1.81e-4, 1.65e-3, 0]
        cbij[2, :] = [1.73e-5, -2.02e-2, 0, 0]
        cbij[3, :] = [-2e-8, 0, 0, 0]

        subruban = [0.1, 750, 8]
        Urban = [0.3, 500, 15]
        DenseUrban = [0.5, 300, 20]
        HighUrban = [0.5, 300, 50]
        a, b, inta_los, inta_Nlos = 0, 0, 0, 0
        if self.choose == 'subruban':
                a, b = self.cal_a_b(subruban, caij, cbij)
                inta_los = 0.1
                inta_Nlos = 21
        elif self.choose == 'Urban':
                a, b = self.cal_a_b(Urban, caij, cbij)
                inta_los = 1
                inta_Nlos = 20
        elif self.choose == 'DenseUrban':
                a, b = self.cal_a_b(DenseUrban, caij, cbij)
                inta_los = 1.6
                inta_Nlos = 23
        elif self.choose == 'HighUrban':
                a, b = self.cal_a_b(HighUrban, caij, cbij)
                inta_los = 2.3
                inta_Nlos = 34

        PLmax = []
        for time in range(0, self.NSP):
                L = math.sqrt((self.placex - self.SPplacex[time]) ** 2 + (self.placey - self.SPplacey[time]) ** 2)
                H = self.placez
                D = math.sqrt((self.placex - self.SPplacex[time]) ** 2 +
                              (self.placey - self.SPplacey[time]) ** 2 +
                              (self.placez) ** 2)
                theta = 180*math.asin(H / D)/math.pi
                Plos = (1 / (1 + a * math.exp(-b * (theta - a))))
                PNlos = 1 - Plos
                PLmax.append(10 * math.log10(D**self.alpha) + self.lossb + Plos * inta_los + PNlos * inta_Nlos)
        return PLmax, D

    def cal_a_b(self, choose, caij, cbij):
            alpha = choose[0]
            belta = choose[1]
            gama = choose[2]
            a = 0
            b = 0
            for j in range(0, 4):
                for i in range(3 - j):
                    a += ((alpha * belta) ** i) * (gama ** j) * caij[i, j]
                    b += ((alpha * belta) ** i) * (gama ** j) * cbij[i, j]
            return a, b
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
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
        ax.set_zlim(0, 150)
        plt.self.show()
    def trajectory(self):
        plt.cla()
        self.tarx.append(self.placex)
        self.tary.append(self.placey)
        plt.scatter(self.tarx, self.tary, c='r')
        SP = plt.scatter(self.SPplacex, self.SPplacey)
        LIN = plt.plot([self.placex, self.SPplacex[self.cline]], [self.placey, self.SPplacey[self.cline]], '--')
        for time in range(self.NSP):
            plt.text(self.SPplacex[time], self.SPplacey[time], str(time) + '-G=' + str(self.G[time]))


    def render(self, goal, mode='human'):
        if self.viewer == None:

            self.viewer = rendering.Viewer(1000, 1000)
            self.cartrans1 = rendering.Transform()
            self.cartrans2 = rendering.Transform()
            circle = rendering.make_circle(10)
            circle.add_attr(self.cartrans2)
            circle.set_color(0, 0, 0.5)
            self.viewer.add_geom(circle)
            for time in range(self.NSP):
                circle = rendering.make_circle(2)
                circle_transform = rendering.Transform(translation=(self.SPplacex[time]+500, self.SPplacey[time]+500))
                circle.add_attr(circle_transform)
                circle.set_color(.5, 0, 0)
                self.viewer.add_geom(circle)

            UAV = rendering.make_circle(4, filled=True)
            # UAV.add_attr(rendering.Transform(translation=(0, 0)))
            UAV.set_color(0, .5, 0)
            UAV.add_attr(self.cartrans1)
            self.viewer.add_geom(UAV)
        self.cartrans2.set_translation(goal[0]*400+500, goal[1]*400+500)
        self.cartrans1.set_translation(self.placex+500, self.placey+500)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def record(self, a, ps, cline, rate, reward, done):
        basic_data = [self.SPplacex,self.SPplacey, self.G]
        data = [self.placex, self.placey, self.v[0], self.v[1]
               , a[0], a[1], self.P, ps, cline, rate/1e6, reward]
        self.data = np.vstack((self.data, data))
        # if done == 1:
        #     sio.savemat("/home/zachary/matlab程序/UAV/2d.mat", self.data)
        #     sio.savemat("/home/zachary/matlab程序/UAV/basic.mat", basic_data)
    def putout(self):
        basic_data = np.vstack((self.SPplacex, self.SPplacey, self.G))
        return basic_data
    def warmaction(self):
        dx = self.SPplacex
        dy = self.SPplacey
        num = np.argmax(self.G)
        aimx, aimy = dx[num] - self.placex, dy[num] - self.placey
        print(self.placex, self.placey)
        norm = np.sqrt(aimx ** 2 + aimy ** 2)
        aimx = aimx / norm
        aimy = aimy / norm
        if np.abs(self.v[0] + aimx * self.delta * self.amax) > self.Vmax:
            aimx = 0
        if np.abs(self.v[1] + aimy * self.delta * self.amax) > self.Vmax:
            aimy = 0
        return np.array([aimx, aimy, 1])





if __name__ == '__main__':
    env = Downlink_2d_v3()
    env.reset()
    rate, snr = env.Rate()
    print(rate/1e6)
    print(snr)
    print(env.G)
    ###########################3
    tarx = [env.placex]
    tary = [env.placey]
    def road(env):
        dx = env.SPplacex
        dy = env.SPplacey
        num = np.argmax(env.G)
        aimx, aimy = dx[num]-env.placex, dy[num]-env.placey
        print(env.placex, env.placey)
        norm = np.sqrt(aimx**2+aimy**2)
        aimx = aimx/norm
        aimy = aimy/norm
        if np.abs(env.v[0] + aimx * env.delta * env.amax) > env.Vmax:
            aimx = 0
        if np.abs(env.v[1] + aimy * env.delta * env.amax) > env.Vmax:
            aimy = 0
        return np.array([aimx, aimy, 1])
    records = []
    recordv = []
    recorda = []
    recorddone = []
    recordcline = []
    recordrate = []
    recordreward = []
    recordG = []
    recordepisode = []
    recordSP = [env.SPplacex,env.SPplacey]
    done = 0
    try:
        for episode in range(1000):
            while done == 0:
                action = road(env)
                S_, reward, done, info = env.step(action)
                records.append([S_[0],S_[1]])
                recordv.append([S_[2],S_[3]])
                recorda.append(action)
                recordreward.append(reward)
                recorddone.append(done)
                recordcline.append(env.cline)
                recordG.append(env.G)
                recordepisode.append(episode)
                print(reward)
                # fig = plt.figure(1)
                # # plt.cla()
                # ax = Axes3D(fig)
                # ax.scatter3D(tarx, tary, 100*np.ones_like(tarx), 'r', marker='*')
                # ax.scatter3D(env.SPplacex, env.SPplacey, np.zeros_like(env.SPplacex))
                # ax.text(env.placex, env.placey, env.placez,
                #         'loc=' + str([env.placex, env.placey, env.placez]) + '\n'
                #         + 'V=' + str(env.v) + '\n' + 'a=' +str([action[0]*30, action[1]*30])
                #         )
                # if env.cline != -1:
                #     ax.plot([env.placex, env.SPplacex[env.cline]], [env.placey, env.SPplacey[env.cline]],
                #             [env.placez, 0], '--')
                #     ax.text((env.placex + env.SPplacex[env.cline]) / 2, (env.placey + env.SPplacey[env.cline]) / 2,
                #             (env.placez + 0) / 2, str(env.rate[env.cline]/1e6))
                #     ax.text(env.SPplacex[env.cline], env.SPplacey[env.cline], 0,
                #             'loc=' + str(env.SPplacex[env.cline]) + str(env.SPplacex[env.cline]) + '\n'
                #             + 'G=' + str(env.G[env.cline]) + '\n')
                # ax.set_xlim(-400, 400)
                # ax.set_ylim(-400, 400)
                # ax.set_zlim(0, 150)
                # plt.pause(1)
    except KeyboardInterrupt:
        sio.savemat('/home/zachary/matlab程序/UAV/warmdata.mat', {'s': records,'v': recordv,'a': recorda,
                                                             'SP': [env.SPplacex, env.SPplacey],
                                                             'cline':recordcline, 'G':recordG, 'episode': recordepisode
                                                          })
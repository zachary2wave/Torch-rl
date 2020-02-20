

import numpy as np
import math
import cvxopt
import  matplotlib.pyplot  as plt

from Raycheak import RTrackinglos,RTracking_reflect
from Single_reflection_verification import RTracking_reflectVerif
from AngleCalculation import AoADoA,AngleReflection
from FreeLoss import Freeloss
from ula import pula
from Fresnel_coefficients import ReflectCoefficient

losFlag=1#标记是否获取Los,单反射和双反射路径
reflectFlag=1
freq=28e9
c=3e8
e=6.5#相对介电常数
lamb=c/freq#波长
NT=128
Tx_xy=np.array([15,15])#设定发射端和接收端的位置信息
Rx_xy=np.array([40,11])
wall1=[[0,0],[10,0],[10,10],[0,10]]
wall2=[[5,20],[20,20],[20,30],[5,30]]
wall3=[[-5,18],[4,18],[4,25],[-5,25]]
wall4=[[15,-5],[50,-5],[50,5],[15,5]]
wall5=[[25,17],[45,17],[45,27],[25,27]]
walls=np.array([wall1,wall2,wall3,wall4,wall5])
num_wall=walls.shape[0]#两个建筑
wallx=[walls[i,:,0] for i in range(walls.shape[0])]
wally=[walls[i,:,1] for i in range(walls.shape[0])]
fig=plt.figure()
for i in range(num_wall):
    plt.fill(wallx[i],wally[i],facecolor='b',alpha=0.5)
plt.show()

wall_normal=[]#求每个线段的法线并存入矩阵中
for i in range(walls.shape[0]):
    normal=[]
    for j in range(4):
        nor=[]
        dx=walls[i][(j+1)%4][0]-walls[i][j][0]
        dy=walls[i][(j+1)%4][1]-walls[i][j][1]
        nor.append(dy/math.sqrt(dx*dx+dy*dy))
        nor.append(-dx / math.sqrt(dx * dx + dy * dy))
        normal.append(nor)
    wall_normal.append(normal)
wall_normal=np.array(wall_normal)

# ula=pula(7,0.5,d=0.5,lamb=1.0)

#计算RX-Tx的距离和矢量
rxtx=Rx_xy-Tx_xy
distRxTx=math.sqrt(rxtx[0] ** 2 + rxtx[1] ** 2)
rxtxUnitVec= rxtx / distRxTx
PLlos=Freeloss(distRxTx,freq)

flaglos=RTrackinglos(walls,Tx_xy,Rx_xy,wall_normal)
if flaglos==1:#有los径，计算Los径对应的到达角和离开角
    DOALOS,AOALOS=AoADoA(rxtxUnitVec)


print("%%%%%%%%LoS Angle%%%%%%%%%%%%%%%%")
print('DOA is ',DOALOS)
print('AOA is ',AOALOS)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#求单反射点##################
#计算tx关于每个墙壁的投影、镜面对称点和可能的单反射路径
wall_proj=np.zeros([num_wall,4,2])
wall_mirror=np.zeros([num_wall,4,2])
intersection=np.zeros([num_wall,4,2])
wall_reflectflag=np.zeros([num_wall,4])#标记Tx关于哪个墙壁的单反射平面没有被遮挡，1表示没有被遮挡，0表示被遮挡
Point_refVerif=np.zeros([num_wall,4,2])#用于验证的不考虑墙面阻塞反射路径的情况
Flag_refVerif=np.zeros([num_wall,4])
reflecAngle=np.zeros([num_wall,4])
DoA=np.zeros([num_wall,4])
AoA=np.zeros([num_wall,4])
H=cvxopt.matrix(range(NT))
vi=cvxopt.matrix(1j)
if flaglos==1:
    steervec=pula(NT, DOALOS, 0.5, lamb)
    delay=math.cos(2*math.pi*distRxTx/lamb)+vi*math.sin(-2*math.pi*distRxTx/lamb)
    H=H+1/PLlos*delay*steervec


for i in range(4):
    aa=walls[:,i,:]-np.tile(Tx_xy,(num_wall,1))
    bb=wall_normal[:,i,:]
    # wallp1=np.tile(np.dot(aa,bb),(2,1))
    for j in range(num_wall):
        dot1=np.dot(aa[j],bb[j])#计算点乘
        v=np.tile(dot1,(1,2))*bb[j]
        wall_proj[j,i,:]=v+Tx_xy
        wall_mirror[j,i,:]=Tx_xy+2*(wall_proj[j,i,:]-Tx_xy)

        #计算单反射径是否存在，如果存在反射点时什么
        wall_reflectflag[j,i],intersection[j,i,:]=RTracking_reflect(walls, wall_mirror[j,i,:], Rx_xy, wall_normal, j, i,Tx_xy)
        Flag_refVerif[j,i],Point_refVerif[j,i,:]=RTracking_reflectVerif(walls, wall_mirror[j,i,:], Rx_xy, wall_normal, j, i)
        if wall_reflectflag[j,i]==1:
            d1=wall_mirror[j,i,:]-Rx_xy
            disreflectRx=math.sqrt(d1[0]**2+d1[1]**2)
            reflecAngle[j,i],DoA[j,i],AoA[j,i]=AngleReflection(intersection[j,i,:], Tx_xy, Rx_xy,wall_normal,j,i)
            Vcoef,Hcoef=ReflectCoefficient(reflecAngle[j,i],e)########################选择哪个呢？水平极化还是垂直极化？
            PL=Freeloss(disreflectRx,freq)
            delaynlos=math.cos(2*math.pi*disreflectRx/lamb)+vi*math.sin(-2*math.pi*disreflectRx/lamb)
            Gain=1/PL*Vcoef*delaynlos#############此处检查一下
            H = H + Gain * pula(NT, DoA[j,i], 0.5, lamb)
# print('projection is :')
# print(wall_proj)
# print('mirror is :')
# print(wall_mirror)
# print('wall is :')
# print(wall_reflectflag)
# print('intersection')
# print(intersection)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print(reflecAngle)
print(DoA)
print(AoA)

# print('%%%%%%%%%%%验证测试结果，不考虑墙面阻塞%%%%%%%%%%%%%%%%%%')
# print('wall is :')
# print(Flag_refVerif)
# print('intersection')
# print(Point_refVerif)

fig=plt.figure()
for i in range(num_wall):
    plt.fill(wallx[i],wally[i],facecolor='b',alpha=0.5)
if flaglos==1:
    x=[Tx_xy[0],Rx_xy[0]]
    y=[Tx_xy[1],Rx_xy[1]]
    plt.plot(x,y,'g:s')
else:
    plt.plot(Tx_xy[0],Tx_xy[1],'s')
    plt.plot(Rx_xy[0],Rx_xy[1],'s')
plt.text(Tx_xy[0],Tx_xy[1],"Tx",size=15,color='b',weight='light')
plt.text(Rx_xy[0],Rx_xy[1],"Rx",size=15,color='b',weight='light')
for i in range(num_wall):
    for j in range(4):
        if wall_reflectflag[i,j]==1:
            x=[Tx_xy[0],intersection[i,j,0],Rx_xy[0]]
            y=[Tx_xy[1],intersection[i,j,1],Rx_xy[1]]
            plt.plot(x,y,'g')

plt.show()


fig1=plt.figure()#画出所有可能的反射射线，注：当
for i in range(num_wall):
    plt.fill(wallx[i],wally[i],facecolor='b',alpha=0.5)
    plt.text((walls[i,0,0]+walls[i,2,0])/2,(walls[i,0,1]+walls[i,2,1])/2,i,size=15,color='b',weight='light')
x=[Tx_xy[0],Rx_xy[0]]
y=[Tx_xy[1],Rx_xy[1]]
plt.plot(x,y,'g:s')
plt.text(Tx_xy[0],Tx_xy[1],"Tx",size=5,color='b',weight='light')
plt.text(Rx_xy[0],Rx_xy[1],"Rx",size=5,color='b',weight='light')
for i in range(num_wall):
    for j in range(4):
        # if Flag_refVerif[i,j]==1:
        x=[Tx_xy[0],Point_refVerif[i,j,0],Rx_xy[0]]
        y=[Tx_xy[1],Point_refVerif[i,j,1],Rx_xy[1]]
        plt.text(Point_refVerif[i,j,0],Point_refVerif[i,j,1],(i,j))
        plt.plot(x,y,'g')
plt.show()


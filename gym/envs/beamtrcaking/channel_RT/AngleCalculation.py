#给定Los径矢量或者是单反射路径(Tx-reflectpoint,reflectpoint-Rx)计算对应的角度信息
#矢量为单位矢量
#Tx角度设定以[1，0]为基准，即与x正半轴的夹角
#Rx角度设定以[1,0]为基准，即与x负半轴的夹角

import numpy as np
import math

def AoADoA(LosUnitVec):#矢量为endpoint减去startpoint，AoA为startpoint处的角度
    DoA=math.degrees(math.acos(LosUnitVec[0]))#[0,180]
    if LosUnitVec[1]<0:
        DoA=-DoA#离开方位角范围设定为(-180,180]
    if LosUnitVec[1]>=0:
        AoA=DoA-180
    else:
        AoA=DoA+180#到达角范围[-180,180)
    return DoA,AoA


def AngleReflection(intersection,Tx,Rx,wall_normal,index1,index2):
    txreflectvec=intersection-Tx
    rxreflectvec=intersection-Rx
    distxreflect = math.sqrt((txreflectvec[0]) ** 2 + (txreflectvec[1]) ** 2)
    disrxreflect=math.sqrt((rxreflectvec[0])**2+(rxreflectvec[1])**2)
    normal=wall_normal[index1,index2,:]
    cos_angle=normal.dot(-rxreflectvec)/disrxreflect
    reflecAngle=math.degrees(math.acos(cos_angle))#角度范围0-90


    DoA=math.degrees(math.acos(txreflectvec[0]/distxreflect))
    if txreflectvec[1]<0:
        DoA=-DoA
    AoA=math.degrees(math.acos(rxreflectvec[0]/disrxreflect))#角度范围（-180,180]
    if rxreflectvec[1]<0:
        AoA=-AoA
    return reflecAngle,DoA,AoA


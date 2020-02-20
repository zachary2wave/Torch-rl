


#计算反射的垂直极化波和水平极化波的反射系数
#参考《cell coverage analysis of 28 GHz millimeter wave in urban microcell environment uing 3-D ray tracking》

import math

def ReflectCoefficient(reflectangle,e):#e为相对介电常数

    Vcoef=(math.cos(reflectangle)-math.sqrt(e-math.sin(reflectangle)**2))\
        /(math.cos(reflectangle)+math.sqrt(e-math.sin(reflectangle)**2))
    Hcoef=(e*math.cos(reflectangle)-math.sqrt(e-math.sin(reflectangle)**2))\
        /(e*math.cos(reflectangle)+math.sqrt(e-math.sin(reflectangle)**2))
    Vcoef=abs(Vcoef)
    Hcoef=abs(Hcoef)

    return Vcoef,Hcoef



# def ReflectCoefficient(reflectangle,e1,e2,f):#e1为媒介相对介电常数，e2为媒质导电率
#     lambda1=3e8/f
#     e=e1-1j*60*e2*lambda1
#     Vcoef=(math.cos(reflectangle)-math.sqrt(e-math.sin(reflectangle)**2))\
#         /(math.cos(reflectangle)+math.sqrt(e-math.sin(reflectangle)**2))
#     Hcoef=(e*math.cos(reflectangle)-math.sqrt(e-math.sin(reflectangle)**2))\
#         /(e*math.cos(reflectangle)+math.sqrt(e-math.sin(reflectangle)**2))
#
#     return Vcoef,Hcoef


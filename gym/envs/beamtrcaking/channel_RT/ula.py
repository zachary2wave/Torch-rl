# Uniform Linear Array

import math
import cvxopt

def pula(n,x,d=0.5,lamb=1.0):
    '''
    Function:
        Generate uniform linear array
    Parameters:
        param n: number of antennas
        param x: sin(theta)
        param d: distance between two adjacent antennas
        param lamb: signal wave length
    Return:
         Array response vector
    '''

    vi=-cvxopt.matrix(1j)
    ind=cvxopt.matrix(range(n))

    arr=1.0/cvxopt.sqrt(n)*cvxopt.exp(vi*2.0*math.pi/lamb*d*x*ind)
    return arr

def pulaext(n,xvec,d=0.5,lamb=1.0):
    '''
    Function:
        Generate uniform linear array
    Parameters:
        param n: number of antennas
        param x: vector of sin(theta)
        param d: distance between two adjacent antennas
        param lamb: signal wave length
    Return:
         Matrix with each column an array response vector
    '''

    vi=-cvxopt.matrix(1j)
    ind=cvxopt.matrix(range(n))

    xlen=len(xvec)
    mat_arr=cvxopt.matrix(0,size=(n,xlen),tc='z')

    for ip in range(xlen):
        mat_arr[:,ip]=1.0/cvxopt.sqrt(n)*cvxopt.exp(vi*2.0*math.pi/lamb*d*xvec[ip]*ind)

    return mat_arr

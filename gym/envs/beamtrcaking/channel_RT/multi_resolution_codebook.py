# Generate Codebook

from envs.beamtrcaking.channel_RT.ula import pulaext, pula
import numpy as np
import math
import cvxopt

def pcodebook(n,s,bshow=None):
    '''
    Function:
        Generate multi-resolution codebook for uniform linear array
    Parameters:
        param n: number of antennas
        param s: size of codebook
    Return:
         Array response vector
    '''

    sample=np.linspace(-1.0,1.0-1.0/s,s)
    cb=pulaext(n,sample)
    wm=1#a new varible wm to minimize the variation in the main-lobe directivity gain
    K=int(math.log2(n)+1)
    F=[]
    vi = -cvxopt.matrix(1j)
    for k in range(1,K+1):
        nn=2**(k-1)#第k级码本的码本个数
        nk=n//nn
        Fk= cvxopt.matrix(0, size=(n, nn), tc='z')
        # Fk=np.zeros([n,nn])
        for i in range(nn):
            for j in range(nk):
                Fk[:,i]+=1/cvxopt.sqrt(nk)*cb[:,i*nk+j]*cvxopt.exp(vi*wm*(i*nk+j+1))

        F.append(Fk)
    return F














    # if bshow==None:
    #     return cb
    # else:
    #     mlen=len(sample)
    #     arr=pulaext(n,sample)
    #     xla=np.linspace(-1.0,1.0,721)
    #     xlen=len(xla)
    #
    #     res=cvxopt.matrix(0,size=(mlen,xlen),tc='d')
    #     for ind_row in range(xlen):
    #         ares=pula(n,xla[ind_row])
    #         for ind_col in range(mlen):
    #             res[ind_col,ind_row]=np.abs(cvxopt.blas.dot(arr[:,ind_col],ares))
    #
    #     import matplotlib.pyplot as plt
    #     ax = plt.subplot(111, projection='polar')
    #
    #     theta=list(xla*np.pi)
    #     for idx_h in range(s):
    #         ax.plot(theta, list(res[idx_h,:]), lw=2,color='magenta')
    #
    #     plt.show()
    #
    #     return cb
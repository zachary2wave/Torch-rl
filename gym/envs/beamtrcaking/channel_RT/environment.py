# Intelligent beam training for millimeter wave communication system via deep reinforcement learning

import numpy as np
import random
import cvxopt
import math

from ibt_cf.plib.codebook import pcodebook
from ibt_cf.ibt_lib.ibt_chan import pibtchan
from ibt_cf.ibt_lib.action_2_beams import pibtaction2beams

vi = cvxopt.matrix(1j)

#######################################################################################################################
# Definition and Implementation of Environment for Single User
#######################################################################################################################
class PPoint2PointSingleUser:
    def __init__(self,ant_num,cb_size,st_bnd,sample_prob,nlos_num=3,T_0=0.001,T_S=0.300):
        self.ant_num=ant_num
        self.cb_size=cb_size

        self.codebook=pcodebook(self.ant_num,self.cb_size)

        self.oseq=np.linspace(st_bnd,st_bnd+len(sample_prob)-1,len(sample_prob))
        self.sample_prob=sample_prob

        self.curr_beam_ind=None
        self.curr_chan=None

        self.nlos_num=nlos_num
        self.T_0=T_0
        self.T_S=T_S

    def reset(self):
        # Initial Optimal Reference Beam
        self.curr_beam_ind = random.sample(np.arange(0, self.cb_size).tolist(), 1)[0]
        # Current Channel
        self.curr_chan=pibtchan(self.ant_num,[-1.0+self.curr_beam_ind*2.0/self.ant_num],self.nlos_num)

        return self.curr_beam_ind

    def natural_evolve(self):
        # Random Sampling
        offset=np.random.choice(self.oseq,p=self.sample_prob)
        self.curr_beam_ind=(self.curr_beam_ind+offset)%self.cb_size

        # Generate Current Channel
        self.curr_chan=pibtchan(self.ant_num,[-1.0+self.curr_beam_ind*2.0/self.cb_size],self.nlos_num)

        return self.curr_beam_ind,self.curr_chan

    def evolve(self,action,trans_pow):
        # Sensing Interval
        sen_beam_inds=np.linspace(start=self.curr_beam_ind+action[0],
                                  stop=self.curr_beam_ind+action[0]+action[1]-1,num=action[1])
        # Sweeping Beams
        swp_beam_inds=[int(round(arg))%self.cb_size for arg in sen_beam_inds]

        # Random Sampling
        offset=np.random.choice(self.oseq,p=self.sample_prob)
        self.curr_beam_ind=(self.curr_beam_ind+offset)%self.cb_size

        # Generate Current Channel
        self.curr_chan=pibtchan(self.ant_num,[-1.0+self.curr_beam_ind*2.0/self.cb_size],self.nlos_num)

        # Is Aligned
        opt_beam_ind=swp_beam_inds[0]
        opt_rece_sig=np.abs(math.sqrt(trans_pow)*cvxopt.blas.dot(self.codebook[:,opt_beam_ind],self.curr_chan)+
                            (cvxopt.normal(1, 1) + vi * cvxopt.normal(1, 1)) / math.sqrt(2.0))

        for swp_ind in swp_beam_inds[1:]:
            test_rece_sig=np.abs(math.sqrt(trans_pow)*cvxopt.blas.dot(self.codebook[:,swp_ind],self.curr_chan)+
                                 (cvxopt.normal(1, 1) + vi * cvxopt.normal(1, 1)) / math.sqrt(2.0))
            if opt_rece_sig<test_rece_sig:
                opt_beam_ind=swp_ind
                opt_rece_sig=test_rece_sig

        is_align=False
        if opt_beam_ind==self.curr_beam_ind:
            is_align=True
            opt_rece_sig_no_noise=math.sqrt(trans_pow)*cvxopt.blas.dot(self.codebook[:,opt_beam_ind],self.curr_chan)
            opt_rewd = np.log2(1 + np.abs(opt_rece_sig_no_noise) ** 2)*(1-action[1]*self.T_0/self.T_S)
        else:
            opt_rece_sig_no_noise=math.sqrt(trans_pow)*cvxopt.blas.dot(self.codebook[:,int(self.curr_beam_ind)],
                                                                       self.curr_chan)
            opt_rewd = np.log2(1 + np.abs(opt_rece_sig_no_noise) ** 2)*(1-self.cb_size*self.T_0/self.T_S)

        return self.curr_beam_ind,opt_rewd,is_align

#######################################################################################################################
# Definition and Implementation of Environment for Multi-User
#######################################################################################################################
class PDownlinkMultiUser:
    def __init__(self,user_num,rfc_num,tran_ante_num,cb_size,cons_imgs_num,next_sbeam):
        self.U=user_num
        self.T=rfc_num
        self.N=tran_ante_num
        self.M=cb_size

        self.cons_imgs_num=cons_imgs_num
        self.next_sbeam=next_sbeam

        self.codebook=pcodebook(self.N,self.M)

        self.curr_beam_inds=None
        self.prev_beam_inds=None
        self.curr_chan=None
        self.curr_state=None
        self.reward=0

        # Start Matlab Engine
        # eng=matlab.engine.start_matlab()

    def __del__(self):
        # Engine Exit
        # eng.quit()
        pass

    def reset(self):
        # Initial Reference Beams in Current Episode
        self.curr_beam_inds = np.array(random.sample(np.arange(1, self.N + 1, 1).tolist(), self.U))
        self.prev_beam_inds=self.curr_beam_inds

        # Initial State
        self.curr_chan=pibtchan(self.N,-1.0+self.curr_beam_inds*2.0/self.N,2)
        hequ = self.codebook.ctrans() * self.curr_chan

        self.curr_state=np.zeros([self.N,self.U*self.cons_imgs_num],np.float)
        hequ_abs=np.array(np.abs(hequ))
        st_ind=self.U*(self.cons_imgs_num-1)
        en_ind=self.U*self.cons_imgs_num
        self.curr_state[:,st_ind:en_ind]=hequ_abs

        return self.curr_state.copy()

    def step(self,action):
        # Beam Indices and Offsets in Next Time-slot
        while True:
            next_beam_inds,offsets=pibtevolve(self.curr_beam_inds,self.next_sbeam,self.M)
            if next_beam_inds[0]!=next_beam_inds[1]:
                break
        self.curr_beam_inds=next_beam_inds

        # Generate Channels in Next Time-slot
        self.curr_chan=pibtchan(self.N,-1.0+self.curr_beam_inds*2.0/self.N,2)

        # Determine Training Beams
        training_beams = pibtaction2beams(self.prev_beam_inds, action, self.M)
        self.prev_beam_inds=self.curr_beam_inds

        # Execute Chosen Action to Determine Current State/Observation
        hequ=cvxopt.matrix(0,size=(self.N,self.U),tc='z')
        len_train_beams=len(training_beams)

        for ind_b in range(len_train_beams):
            trans_patt=self.codebook[:,training_beams[ind_b].tolist()-1]
            hequ[training_beams[ind_b].tolist()-1,:]=trans_patt.ctrans()*self.curr_chan

        st_ind=self.U*(self.cons_imgs_num-1)
        en_ind=self.U*self.cons_imgs_num
        temp=self.curr_state[:,st_ind:en_ind]
        self.curr_state[:, :self.U * (self.cons_imgs_num - 1)]=temp
        hequ_abs=np.array(np.abs(hequ))
        self.curr_state[:,st_ind:en_ind]=hequ_abs

        b_align = self.curr_beam_inds[0] in training_beams and self.curr_beam_inds[1] in training_beams
        if not b_align:
            ratio=1-(t_S*self.M+t_P)/t_C
            return self.curr_state.copy(),ratio*sta_sum_rate[0],True,{}

        # Binary Selection Matrix
        B=cvxopt.matrix(0, size=(self.M,self.U), tc='d')
        B[self.curr_beam_inds[0].tolist()-1,0]=1
        B[self.curr_beam_inds[1].tolist()-1,1]=1

        hb=B.ctrans()*hequ

        mhb=np.array(hb)
        # sum_rate=eng.py_dig_prec_vec(matlab.double(mhb.real.tolist()),matlab.double(mhb.imag.tolist()),idx_snr,nargout=1)
        ratio=1-(t_S*len_train_beams+t_P)/t_C
        self.reward=ratio

        return self.curr_state.copy(),self.reward,False,{}

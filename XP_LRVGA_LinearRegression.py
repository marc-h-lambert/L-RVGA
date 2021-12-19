###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper Section 7.2 Linear regression                #                                                                            
# "The limited memory recursive variational Gaussian approximation (L-RVGA)"      #                  
# Authors: Marc Lambert, Silvere Bonnabel and Francis Bach                        #
###################################################################################

import numpy as np
from KalmanMachine.KDataGenerator import LinearRegObservations
from KalmanMachine.Kalman4LinearReg import LKFLinReg, LargeScaleLKFLinReg
from KalmanMachine.KEvalPosterior import PosteriorLinReg
from KalmanMachine.KVizualizationsHD import plotLSKLlinReg_lattent
import matplotlib.pyplot as plt
from plot4latex import set_size
import math
import time, tracemalloc
from matplotlib.ticker import (MultipleLocator,AutoMinorLocator,LogLocator,MaxNLocator)

# SENSITIVITY TO THE DIMENSION OF THE LATTENT SPACE (p)
def XP_HighDim_LinReg_Lattent(axs,sigma0,mu0,sigmaNoise,N,d,list_p,c,ppca,svd,nbInnerLoop,seed,label=True):
    ################### GENERATE DATA ####################### 
    RegObs=LinearRegObservations(sigmaNoise,N,d,c,seed,scale=1,rotate=True,normalize=True)
    y,X=RegObs.datas
    
    ################### GROUND TRUTH ####################### 
    theta0=mu0*np.ones([d,1])/math.sqrt(d) 
    Cov0=np.identity(d)*sigma0**2
    posterior=PosteriorLinReg(theta0,Cov0).fit(X,y.reshape(N,),sigmaNoise)
    
    tic=time.perf_counter()
    lkf = LKFLinReg(theta0,Cov0,sigma=1,passNumber=1).fit(X, y.reshape(N,),monitor=True)
    toc=time.perf_counter()
    
    lkf_perfo_time=[]
    lkf_perfo_memory=[]
    lkf_perfo_time.append(lkf.timePerIteration)
    lkf_perfo_memory.append(lkf.memoryUsed)
    print("Run LKF in={0:.2}s".format(toc-tic))
    print("Memory cost for LKF is {0} MB;".format(lkf.memoryUsed))
    print("Time per iteration for LKF is {0} s;".format(lkf.timePerIteration))
    print("The KL divergence for LKF is {0}".format(posterior.divergence(lkf.theta,lkf.Cov)))
    
    ################### INITIALIZATION  & RUN LARGE SCALE KALMAN ####################### 
    std0=1/sigma0   
    psi0=std0**2*np.ones([d,1])

    list_lskf=[]
    for p in list_p:
        np.random.seed(seed)
        if p==0 or svd:
            B0=np.zeros([d,p])
        else:
            B0=1e-2*np.random.multivariate_normal(np.zeros(p),np.identity(p),(d))

        tic=time.perf_counter()
        lskf = LargeScaleLKFLinReg(theta0, psi0, B0, passNumber=1, sigma=1, ppca=ppca, svd=False, \
                                   nbInnerLoop=nbInnerLoop).fit(X, y.reshape(N,),monitor=True)
        toc=time.perf_counter()

        print("Run LSKF (p={0}) with EM in={1:.2}s".format(p,toc-tic))
        print("Memory cost for LSKF (p={0})  is {1} MB;".format(p,lskf.memoryUsed))
        print("Time per iteration for LSKF (p={0}) is {1} s;".format(p,lskf.timePerIteration))
        print("The KL divergence for LSKF (p={0}) is {1}".format(p,posterior.divergence(lskf.theta,lskf.Cov)))
        lkf_perfo_time.append(lskf.timePerIteration)
        lkf_perfo_memory.append(lskf.memoryUsed)
    
        list_lskf.append(lskf)
        
    plotLSKLlinReg_lattent(axs[0],lkf,list_lskf,list_p,posterior,label)
    return lkf_perfo_time,lkf_perfo_memory
    
if __name__=="__main__":
    Test=["LinHD2"] # change the label to change the Section tested
    num=1    
    
    # LINEAR REGRESSION: test on inner loops
    if 'LinHD1' in Test:
        print("######### LinHD1 : Sensitivity to number of inner loops ##############")
        sigma0=1
        mu0=0
        d=100
        list_p=[d]
        N=1000
        c=1
        seed=1
        ppca=False
        svd=False
        nbInnerLoop_list=[1,2,3]
        sigmaNoise=1
        
        fig, axs = plt.subplots(1, 3, sharey=True,sharex=True,figsize=set_size(ratio=0.5),num=num)
        
        i=0
        label=True
        for ax in axs:
            XP_HighDim_LinReg_Lattent(np.array([ax]),sigma0,mu0,sigmaNoise,N,d,list_p,c,ppca,svd,nbInnerLoop_list[i],seed,label=True)
            ax.set_title(r'Nb inner loops = {}'.format(nbInnerLoop_list[i]))
            ax.legend(loc="upper right")
            ax.xaxis.set_major_locator(MultipleLocator(300))
            ax.yaxis.set_major_locator(MultipleLocator(500))
            if i==0:
                ax.set_ylabel('KL error')
            ax.set_xlabel('number of iterations')
            i=i+1
                
        plt.savefig('./outputs/LSLinReg_innersLoops')   
        num=num+1
    
    # LINEAR REGRESSION: test on p
    # We have reduced the dimension to speed up, to find exactely paper result 
    # put d=1000 and N=3000
    if 'LinHD2' in Test:
         # The recusrive EM converge to the linear Kalman filter for p higher enough 
        print("######### LinHD2 : Sensitivity to lattent dimension p ##############")
        d=100 #1000
        N=1000 #3000
        nbInnerLoop=3 
        
        sigma0=1
        mu0=0
        sigmaNoise=1
        list_p=[100,10,2,1]
        c=1
        seed=1
        ppca=False
        svd=False                 
        
        fig, ax = plt.subplots(1, 1, sharex=False,figsize=set_size(ratio=1,fraction=0.7),num=num)
        lkf_perfo_time,lkf_perfo_memory=XP_HighDim_LinReg_Lattent(np.array([ax]),sigma0,mu0,sigmaNoise,N,d,list_p,c,ppca,svd,nbInnerLoop,seed,label=True)
        ax.set_title("$N={0}$, $d={1}$, $InLoop={2}$".format(N,d,nbInnerLoop))
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.yaxis.set_major_locator(LogLocator(numticks=4))
        ax.grid()
        ax.legend(loc="lower left", ncol=3)
        ax.set_xlabel('number of iterations')
        ax.set_ylabel('KL')
        plt.tight_layout()
        plt.savefig('./outputs/LRVGA_LinReg_KL_{0}_{1}_{2}'.format(N,d,nbInnerLoop))
        num=num+1
###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper Section 7.1 Covariance matrix approximation  #                                                                                                  
# "The limited memory recursive variational Gaussian approximation (L-RVGA)"      #                  
# Authors: Marc Lambert, Silvere Bonnabel and Francis Bach                        #
###################################################################################

import numpy as np

from KalmanMachine.KFactorAnalysis import CovarianceFactorAnalysisRecEM,\
    CovarianceFactorAnalysisEM_batch, CovarianceFactorAnalysisOnlineEM
from KalmanMachine.KUtils import  KLDivergence, KLDivergence_largeScale
import matplotlib.pyplot as plt
from plot4latex import set_size
import numpy.linalg as LA
import time, tracemalloc
import numpy.random
import math
from matplotlib.ticker import (MultipleLocator,AutoMinorLocator,LogLocator,MaxNLocator)

def read_LibSVM(list_y,list_x,reformatY=True,rescaleX=False):
    N=len(list_y)
    d=len(list_x[0])
    tab=pd.DataFrame(list_x).fillna(0).values
    X=tab[:,0:d]
    y=pd.DataFrame(list_y).values

    if reformatY:
        y=(y+1)/2
    if rescaleX:
        Xnorms=LA.norm(X,axis=1)
        k=np.mean(Xnorms)
        X=X/k
    y=y.reshape(N,)
    X=X.reshape(N,d)
    
    return y,X

def LargeScaleCovariance(ax,X,N,d,p,numPass,nbInnerLoopRecEM,nbLoopBatchEM,showBatchEM=True,showFixedPointEM=False):
    S=X.T.dot(X)/N
    (sign, logdetS) = LA.slogdet(S)
    print("Det(S)=",LA.det(S))
    print("logdetS=",logdetS)

    ############# Initialization ###########
    # The inputs have been previousely normalized such that sigma=1 #
    ppca=False
    epsilon=1e-8
    sigma0=math.sqrt((1-epsilon)/d)
    psi0=sigma0**2*np.ones([d,1])
    W0=np.random.multivariate_normal(np.zeros(p),np.identity(p),(d))
    for i in range(0,p):
        W0[:,i]=W0[:,i]/LA.norm(W0[:,i])
    sw0=math.sqrt(epsilon/p)
    W0=W0*sw0
    
    #S0=np.diag(psi0.reshape(d,))+W0.dot(W0.T)
    kl0=KLDivergence_largeScale(S,logdetS,psi0,W0)#KLDivergence(S,S0)
    print("The inital KL divergence is: ",kl0)
    
    #idx=1#int(N/10)
    
    
    ############# Online EM ###########
    OnlineEm=CovarianceFactorAnalysisOnlineEM(psi0, W0, ppca=ppca,nbInnerLoop=1,fixedPoint=False,averaging=True)
    KLonlinefa=np.zeros([numPass*N+1,])
    KLonlinefa[0]=kl0
    timeSpan=0
    Xo=X
    for npass in range(0,numPass):
        if npass>0:
            np.random.shuffle(Xo)

        for t in range(0,N): 
            tic = time.perf_counter()
            if t==1 or t==10:
                 tracemalloc.start()
            
            xt=Xo[t].T.reshape(d,1)
            OnlineEm.fit(xt,npass*N+t)
        
            toc = time.perf_counter()
            timeSpan+=toc-tic
            if t==1 or t==10:
                 current, peak = tracemalloc.get_traced_memory()
                 tracemalloc.stop()
                 print('Compute inner loop Online EM at t={0} in {1:.2}'.format(t,toc-tic))
                 print("Memory usage for Online EM at t={0} is {1} MB; Peak was {2} MB".format(t,current / 10**6,peak / 10**6))
            KLonlinefa[npass*N+t+1]=KLDivergence_largeScale(S,logdetS,OnlineEm.psi,OnlineEm.W)#KLDivergence(S,OnlineEm.faCov)
    
    for i in range(0,N):
        if KLonlinefa[i]<kl0:
            idx=i
            break
    ax.plot(range(idx,KLonlinefa.shape[0]),KLonlinefa[idx:],label="Online-EM",color='red',linewidth=1.5)     
    print('Compute Online EM... in {0:.2}'.format(timeSpan))
    #print("Current memory usage for OnlineEM is {0} MB; Peak was {1} MB".format(current / 10**6,peak / 10**6))
    print("The KL divergence after stochastic EM is: ",KLonlinefa[-1])
    
    ############# Recusrive EM ###########
    RecEm=CovarianceFactorAnalysisRecEM(psi0, W0, ppca=ppca,nbInnerLoop=nbInnerLoopRecEM,fixedPoint=False)
    KLrecfa=np.zeros([numPass*N+1,])
    KLrecfa[0]=kl0
    timeSpan=0
    for npass in range(0,numPass):
        for t in range(0,N): 
            tic = time.perf_counter()
            if t==1 or t==10:
                tracemalloc.start()
            
            xt=X[t].T.reshape(d,1)
            if t==0 and npass==1:
                firstInput=True
            else:
                firstInput=False
            RecEm.fit(xt.T,npass*N+t,firstInput=firstInput)
            
            toc = time.perf_counter()
            timeSpan+=toc-tic
                        
            if t==1 or t==10:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                print('Compute inner loop Recusrive EM at t={0} in {1:.2}'.format(t,toc-tic))
                print("Memory usage for RecEM at t={0} is {1} MB; Peak was {2} MB".format(t,current / 10**6,peak / 10**6))
            
            KLrecfa[npass*N+t+1]=KLDivergence_largeScale(S,logdetS,RecEm.psi,RecEm.W)#KLDivergence(S,RecEm.faCov)
    #print(KLrecfa)
    for i in range(0,N):
        if KLrecfa[i]<kl0:
            idx=i
            break
    ax.plot(range(idx,KLrecfa.shape[0]),KLrecfa[idx:],label="Recursive-EM (Ours)",linestyle="dashed",color='green',linewidth=1.5)
    print('Compute Recusrive EM... in {0:.2}'.format(timeSpan))
    print("The KL divergence after weighted recursive EM is: ",KLrecfa[-1])
    
      ############# Batch EM ###########
    if showBatchEM:
        # s0=1/d
        # psi0=s0*np.ones([d,1])
        # W0=math.sqrt(s0)*np.random.multivariate_normal(np.zeros(p),np.identity(p),(d))
        BatchEm=CovarianceFactorAnalysisEM_batch(psi0, W0, ppca=ppca,nbInnerLoop=nbLoopBatchEM,fixedPoint=False)
        KLbatchfa=np.zeros([N*BatchEm._nbInnerLoop+2,])
        KLbatchfa[0]=kl0
        timeSpan=0
        for t in range(0,BatchEm._nbInnerLoop+1):  
            tic = time.perf_counter()
            if t==1 or t==10:
                  tracemalloc.start()
            
            KLbatchfa[t*N+1:(t+1)*N+1]=KLDivergence_largeScale(S,logdetS,BatchEm.psi,BatchEm.W)#KLDivergence(S,BatchEm.faCov)
            psiFA,WFA=BatchEm.fit(S,p)
            
            toc = time.perf_counter()
            timeSpan+=toc-tic
            
            if t==1 or t==10:
                  current, peak = tracemalloc.get_traced_memory()
                  tracemalloc.stop()
                  print('Compute loop Batch EM at t={0} in {1:.2}'.format(t,toc-tic))
                  print("Memory usage for Batch EM at t={0} is {1} MB; Peak was {2} MB".format(t,current / 10**6,peak / 10**6))
            
            
        ax.plot(range(idx,KLbatchfa.shape[0]),KLbatchfa[idx:],label="Batch-EM",color='blue',linewidth=2)   
        print('Compute Batch EM... in {0:.2}'.format(timeSpan))
        print("The KL divergence after batch EM is: ",KLbatchfa[-1])
    
    # ############# FixPoint EM ###########
    if showFixedPointEM:
        FixPointEm=CovarianceFactorAnalysisEM_batch(psi0, W0, ppca=ppca,nbInnerLoop=1,fixedPoint=False)
        KLfpfa=np.zeros([N+1,])
        KLfpfa[0]=kl0
        St=np.zeros([d,d])
        timeSpan=0
        for t in range(0,N): 
            tic = time.perf_counter()
            if t==0:
                tracemalloc.start()
                
            xt=X[t].T.reshape(d,1)
            St=(t/(t+1))*S+xt.dot(xt.T)/(t+1)
            FixPointEm.fit(St,p)
            
            if t==0:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            toc = time.perf_counter()
            timeSpan+=toc-tic
            
            KLfpfa[t+1]=KLDivergence(S,FixPointEm.faCov)
        
        ax.plot(range(idx,KLfpfa.shape[0]),KLfpfa[idx:],label="FixedPoint-EM") 
        print('Compute Fixed Point EM... in {0:.2}'.format(timeSpan))
        print("Current memory usage for Fixed Point EM is {0} MB; Peak was {1} MB".format(current / 10**6,peak / 10**6))
        print("The KL divergence after FixedPoint EM is: ",KLDivergence(S,FixPointEm.faCov))

if __name__=="__main__":
    Test=["RecursiveEMvsOnlineEM"] # change the label to change the Section tested
    num=1    
    
    # Factorization of random covariance matrix        
    if 'RecursiveEMvsOnlineEM' in Test:
        print("##########################  Recursive EM vs Online EM on random data Set ###################")
        d=1000
        p=10
        N=10000
        nbInnerLoopRecEM=3
        nbLoopBatchEM=10
        numPass=1
        seed=1
        normalize=True
        ppca=False
        np.random.seed(seed)

        # the matrix we want to find
        np.random.seed(seed)
        psi=np.random.uniform(size=[d,1])
        np.random.seed(seed)
        W=np.random.multivariate_normal(np.zeros(p),np.identity(p),(d))
        Cov=np.diag(psi.reshape(d,))+W.dot(W.T)
        # we sample N Gaussian random variables from the Cov matrix 
        np.random.seed(seed)
        X=np.random.multivariate_normal(np.zeros(d),Cov,(N))
     
        if normalize:
            for i in range(0,N):
                X[i]=X[i]/LA.norm(X[i])
        
        num=num+1
        fig, (ax) = plt.subplots(1, 1,figsize=set_size(fraction=0.6,ratio=0.7,twoColumns=True), sharex=False,num=num)
        LargeScaleCovariance(ax,X,N,d,p,numPass,nbInnerLoopRecEM,nbLoopBatchEM,showBatchEM=True,showFixedPointEM=False)
        ax.set_title('$d={}, p={}, N={}$ \n Recursive EM inner loop $={}$'.format(d,p,N,nbInnerLoopRecEM))
        ax.set_xlabel('number of data processed')
        ax.set_ylabel('KL')
        #ax.set_yscale('log')
        ax.grid(True)
        #ax.xaxis.set_major_locator(MultipleLocator(10*int(N/10)))
        ax.legend(loc="upper right", ncol=1)

        plt.tight_layout()
        plt.savefig('./outputs/LargeScaleCov_RAND_d{0}_p{1}_N{2}_loop{3}_numPass{4}'.format(d,p,N,nbInnerLoopRecEM,numPass))

        plt.show()
        
     # Factorization of LibSVM covariance matrix  
     # Require  LibSVM package (and panda): "pip install libsvm"   
    if 'RecursiveEMvsOnlineEM-LibSVM' in Test:
        from libsvm.svmutil import *
        import pandas as pd
        print("##########################  Recursive EM vs Online EM on LIBSVM data Set ###################")
        dataset="Madelon"
        # uncomment the desired line to change the dataset
        list_y, list_x = svm_read_problem("./DataSet/Madelon/madelon.txt")
        #list_y, list_x = svm_read_problem("./DataSet/BreastCancer/breast-cancer_scale.txt") 
        #list_y, list_x = svm_read_problem("./DataSet/Heart/heart_scale.txt")
        
        Y,X=read_LibSVM(list_y,list_x,rescaleX=True)
        
        N=len(list_y)
        d=len(list_x[0])
        print("N=",N)
        print("d=",d)
        p=20
        nbInnerLoopRecEM=3
        nbLoopBatchEM=5
        numPass=1
        seed=1
        normalize=True
        ppca=False
        np.random.seed(seed)
        
        num=num+1
        fig, ax = plt.subplots(1, 1,figsize=set_size(fraction=0.5,ratio=0.7,twoColumns=True), sharex=False,num=num)
        LargeScaleCovariance(ax,X,N,d,p,numPass,nbInnerLoopRecEM,nbLoopBatchEM,showBatchEM=True,showFixedPointEM=False)
        
        ax.set_title('{} dataset $d={}, p={}, N={}$ \n Recursive EM inner loop $={}$'.format(dataset,d,p,N,nbInnerLoopRecEM))
        ax.set_xlabel('number of iterations')
        ax.set_ylabel('KL')
        ax.set_yscale('log')
        ax.legend(loc="upper right", ncol=1)
        ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(100*int(N/100)))
        plt.tight_layout()
        plt.savefig('./outputs/LargeScaleCov_{0}_d{1}_p{2}_N{3}_loop{4}_numPass{5}'.format(dataset,d,p,N,nbInnerLoopRecEM,numPass))

        plt.show()
    
    
    
    
        
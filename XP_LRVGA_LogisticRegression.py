###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper Section 7.3 and 7.4 Logistic regression      #                                                                                    
# "The limited memory recursive variational Gaussian approximation (L-RVGA)"      #                  
# Authors: Marc Lambert, Silvere Bonnabel and Francis Bach                        #
###################################################################################

import numpy as np
from KalmanMachine.KDataGenerator import LogisticRegObservations
from KalmanMachine.Kalman4LogisticReg import RVGALogReg, EKFLogReg, \
    RVGALogRegIterated, RVGALogRegExplicit
from KalmanMachine.Kalman4LogisticRegLS import LargeScaleEKFLogReg, LargeScaleRVGALogReg, \
    LargeScaleRVGALogRegExplicit, LargeScaleRVGALogRegSampled, LargeScaleRVGALogRegSampled2
from KalmanMachine.KEvalPosterior import PosteriorLogReg
from KalmanMachine.KVizualizationsHD import plotLSKLlogReg, plotLSerrorMaplogReg, plotCov
from KalmanMachine.KUtils import  fastLogDet
import matplotlib.pyplot as plt
from plot4latex import set_size
import math
import numpy.linalg as LA
import time, tracemalloc

from matplotlib.ticker import (MultipleLocator,AutoMinorLocator,LogLocator,MaxNLocator,LinearLocator)

    
# Run the baseline implicit RVGA algorithm for different values of the lattent dimension p
def XP_HighDim_LogReg_Lattent(axs,sigma0,mu0,N,d,list_p,c,ppca,svd,nbInnerLoop,seed,\
                              label=True,coef_s=0.2,computeLaplace=True,nbSamplesKL=300,loss=np.array(["kl"])):
    ################### GENERATE DATA ####################### 
    s=1/d**(coef_s) # well separable
    RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
    y,X=RegObs.datas
    
    ################### GROUND TRUTH ####################### 
    theta0=mu0*np.ones([d,1])/math.sqrt(d)
    psi0=np.ones([d,1])/sigma0**2
    W0_prior=np.zeros([d,1])
    posterior=PosteriorLogReg(theta0,psi0,W0_prior,computeLaplace).fit(X,y.reshape(N,),monitor=True) 
    np.random.seed(seed)
    normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamplesKL,))
    if computeLaplace:
        kl_lap=posterior.KL_Laplace(normalSamples)
        print("The KL divergence for Laplace is {0}".format(kl_lap))
        
    ################### INITIALIZATION  & RUN LARGE SCALE KALMAN #######################    
    stdW0=1e-3
    
    list_lskf=[]
    list_labels=[]
    list_methods=[]
    for p in list_p:
        np.random.seed(seed)
        if p==0 or svd:
            B0=np.zeros([d,p])
        else:
            B0=stdW0*np.random.multivariate_normal(np.zeros(p),np.identity(p),(d))
        lodetP0=fastLogDet(psi0,B0)
                
        tic=time.perf_counter()
        lskf = LargeScaleRVGALogReg(theta0, psi0, B0, passNumber=1, ppca=ppca, svd=svd, \
                                      nbInnerLoop=nbInnerLoop).fit(X, y.reshape(N,),monitor=True)
        toc=time.perf_counter()

        print("Run LSKF (p={0}) with EM in={1:.2}s".format(p,toc-tic))
        print("Memory cost for LSKF (p={0})  is {1} MB;".format(p,lskf.memoryUsed))
        print("Time per iteration for LSKF (p={0}) is {1} s;".format(p,lskf.timePerIteration))
        print("The KL divergence for LSKF (p={0}) is {1}".format(p,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
         
        list_lskf.append(lskf)
        list_labels.append('p={}'.format(p))
        list_methods.append("RVGA")
    
    i=0
    labelize=True
    for ax in axs:
        if i>0:
            labelize=False  
        title=''
        if loss[i]=="kl":
            title="KL divergence"
            plotLSKLlogReg(ax,list_lskf,list_methods,list_labels,posterior,labelize,seed,computeLaplace,\
                           nbSamplesKL=nbSamplesKL)
        elif loss[i]=="distanceMAP":
            title="distance to MAP"
            plotLSerrorMaplogReg(ax,list_lskf,list_methods,list_labels,posterior,labelize=labelize)
        
        elif loss[i]=="angleMAP":
            title="angle to MAP"
            plotLSerrorMaplogRegAngles(ax,list_lskf,list_methods,list_labels,posterior,labelize=labelize)
        ax.set_title(title)
        i=i+1
        
# Run some versions of the RVGA (implicit, explicite, iterate, etc..) : can plot the KL divergence or the distance to MAP
def XP_HighDim_LogReg_Method(axs,list_methods,list_labels,sigma0,mu0,N,d,p,c,ppca,svd,nbInnerLoop,seed,\
                             labelize=True,coef_s=0.2,computeLaplace=True,nbSamplesKL=300,loss=np.array(["kl"]),list_col0=[]):
    if svd:
        methodFA='SVD'
    else:
        methodFA='EM'
    if ppca:
        typeFA='PPCA'
    else:
        typeFA='FA'
        
    ################### GENERATE DATA ####################### 
    s=1/d**(coef_s) # well separable
    RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
    y,X=RegObs.datas

    ################### INITIALIZATION  ####################### 
    theta0=mu0*np.ones([d,1])/math.sqrt(d)
    psi0=np.ones([d,1])/sigma0**2
    
    stdW0=1e-2
    np.random.seed(seed)
    if p==0 or svd:
        W0=np.zeros([d,p])
    else:
        W0=stdW0*np.random.multivariate_normal(np.zeros(p),np.identity(p),(d))
    
    ################### GROUND TRUTH ####################### 
    W0_prior=np.zeros([d,d])
    posterior=PosteriorLogReg(theta0,psi0, W0_prior,computeLaplace).fit(X,y.reshape(N,),monitor=True) 
    np.random.seed(seed)
    normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamplesKL,))
    if computeLaplace:
        kl_lap=posterior.KL_Laplace(normalSamples)
        print("The KL divergence for Laplace is {0}".format(kl_lap))
        
    ################### RUN LARGE SCALE KALMAN #######################    

    list_lskf=[]    
    list_col=[]
    for label in list_methods:
        lskf=None
        col='k'
        
        if label=="RVGA":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogReg(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='k'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        elif label=="RVGA-explicit":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegExplicit(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=False).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='b'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        elif label=="EKF":
            tic=time.perf_counter()
            lskf = LargeScaleEKFLogReg(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='dimgrey'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        
        elif label=="RVGA-extragrad":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegExplicit(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=True).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        elif label=="Full-RVGA":
            tic=time.perf_counter()
            Cov0=LA.inv(W0.dot(W0.T)+np.diag(psi0.reshape(d,)))
            lskf = RVGALogReg(theta0, Cov0).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        elif label=="Full-RVGA-explicit":
            tic=time.perf_counter()
            Cov0=LA.inv(W0.dot(W0.T)+np.diag(psi0.reshape(d,)))
            lskf = RVGALogRegExplicit(theta0, Cov0).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        
        
        elif label=="Full-EKF":
            tic=time.perf_counter()
            Cov0=LA.inv(W0.dot(W0.T)+np.diag(psi0.reshape(d,)))
            lskf = EKFLogReg(theta0, Cov0).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        
        elif label=="Full-RVGA-mirrorProx":
            tic=time.perf_counter()
            Cov0=LA.inv(W0.dot(W0.T)+np.diag(psi0.reshape(d,)))
            lskf = RVGALogRegIterated(theta0, Cov0).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        elif label=="Full-RVGA-Partial-mirrorProx":
            tic=time.perf_counter()
            Cov0=LA.inv(W0.dot(W0.T)+np.diag(psi0.reshape(d,)))
            lskf = RVGALogRegIterated(theta0, Cov0,updateCovTwoTimes=False).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='r'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        
        elif label=="RVGA-sampled-1":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=False,nbSamples=1).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='k'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
            
        elif label=="RVGA-sampled-10":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=False,nbSamples=10).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='k'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
            
        elif label=="RVGA-sampled-100":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=False,nbSamples=100).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='k'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        elif label=="RVGA-sampled-extragrad-1":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=True,nbSamples=1).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
            
        elif label=="RVGA-sampled-extragrad-3":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=True,nbSamples=3).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
        
        elif label=="RVGA-sampled-extragrad-10":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=True,nbSamples=10).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
            
        elif label=="RVGA-sampled-extragrad-100":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=True,nbSamples=100).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='g'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
            
        elif label=="RVGA-sampled-extragrad2-100":
            tic=time.perf_counter()
            lskf = LargeScaleRVGALogRegSampled2(theta0, psi0, W0, passNumber=1, ppca=ppca, svd=svd, \
                                  nbInnerLoop=nbInnerLoop,extragrad=True,updateCovTwoTimes=True,nbSamples=100).fit(X, y.reshape(N,),monitor=True)
            toc=time.perf_counter()
            col='r'

            print("Run LSKF ({0}) with {1}-{2} in={3:.2}s".format(label,typeFA,methodFA,toc-tic))
            print("Memory cost for LSKF ({0})  is {1} MB;".format(label,lskf.memoryUsed))
            print("Time per iteration for LSKF ({0}) is {1} s;".format(label,lskf.timePerIteration))
            print("The KL divergence for LSKF ({0}) is {1}".format(label,posterior.divergence(lskf.theta,lskf.Cov,normalSamples)))
         
        
        else:
            print("Name {} is unkown !!".format(label))
        
        if lskf != None:
            list_lskf.append(lskf)
            list_col.append(col)
    
    # we rather take the list of colors in parameter by default 
    if len(list_col0) > 0:
        list_col=list_col0
        
    i=0
    for ax in axs:
        if i>0:
            labelize=False
        title=''
        if loss[i]=="kl":
            plotLSKLlogReg(ax,list_lskf,list_methods,list_labels,posterior,labelize,seed,computeLaplace,\
                           nbSamplesKL=nbSamplesKL,list_col=list_col)
        elif loss[i]=="distanceMAP":
            plotLSerrorMaplogReg(ax,list_lskf,list_labels,posterior,labelize=labelize,list_col=list_col)
        i=i+1
        

# Run XP_HighDim_LogReg_Method for different values of hyperparams: sigma0,inner loop,...
def XP_HighDim_LogReg_DataSET(axs,list_methods,list_labels,sigma0_list,mu0,N,d_list,p_list,\
                 c,ppca,svd,nbInnerLoop_list,seed,coef_s=0.2,computeLaplace=False,\
                 nbSamplesKL=300,loss="kl",list_col0=[]):
        
    i=0
    labelize=True
    for ax in axs:
        if i>0:
            labelize=False
        XP_HighDim_LogReg_Method(np.array([ax]),list_methods,list_labels,sigma0_list[i],mu0,N,d_list[i],p_list[i],c,\
                             ppca,svd,nbInnerLoop_list[i],seed,labelize=labelize,coef_s=coef_s,\
                                 computeLaplace=True,nbSamplesKL=nbSamplesKL,loss=np.array([loss]),list_col0=list_col0)
        ax.set_title(r'$\sigma_0$={0}'.format(sigma0_list[i]))
        ax.set_xlabel('nb iterations')
        if loss=="kl":
            ax.set_ylabel('KL')
        if loss=="distanceMAP":
            ax.set_ylabel('distance to the MAP')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        i=i+1

# Draw the covariance matrix and the true posterior in 2D
def XP_2D(axs,sigma0,mu0,N,s,c,seed,nbLevels=1):
    d=2
    ################### GENERATE DATA ####################### 
    RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
    y,X=RegObs.datas
     
    ################### RUN KALMAN ####################### 
    theta0=mu0*np.ones([d,1])/math.sqrt(d)
    Cov0=np.identity(d)*sigma0**2
    psi0=np.ones([d,1])/sigma0**2
    W0=np.zeros([d,1])
    
    # True posterior
    posterior=PosteriorLogReg(theta0,psi0,W0).fit(X,y.reshape(N,))
    
    # Kalman algorithms
    naturalGrad = EKFLogReg(theta0, Cov0).fit(X, y.reshape(N,),monitor=True)
    rvgaImplicit = RVGALogReg(theta0, Cov0).fit(X, y.reshape(N,),monitor=True)
    rvgaExplicit = RVGALogRegExplicit(theta0,Cov0).fit(X, y.reshape(N,),monitor=True)
    extragrad = RVGALogRegIterated(theta0, Cov0).fit(X, y.reshape(N,),monitor=True)
    
    
    list_lkf=np.array([naturalGrad,rvgaExplicit,rvgaImplicit,extragrad])
    list_labels=np.array(["natural gradient","rvga-explicit","rvga-implicit","rvga-mirrorProx"])
    plotCov(axs,list_lkf,list_labels,posterior,nbLevels)
    
if __name__=="__main__":
    Test=["LogHD"] # change the label to change the Section tested
    
    num=1    
    
    ########################################################################################"
    #  EXPERIMENTS FOR LOGISTIC REGRESSION (Section 5.3)
    ########################################################################################
    
    # We have reduced the dimension to speed up, to find exactely paper result 
    # put d=1000, N=10000 and coef_s=0.26
    if 'LogHD' in Test:
        # The recursive EM converge to the batch Laplace for p higher enough
        print("######### LogHD : Sensitivity to lattent dimension p ##############")
        N=1000
        d=200
        nbInnerLoop=1
        coef_s=0.2#0.26 # drive the separation distance of the two clusters to classify (must change with d)
        
        list_p=[100,10,2,1] # dimension of the lattent space
        sigma0=4 # standard deviation of the prior (supposed isotropic)
        mu0=0 # initial guess for the mean
        c=1 # parameter driving the condition number of the covariance matrix of inputs
        seed=1
        ppca=False
        svd=False
        nbSamplesKL=5 # to compute the unnormalized KL, 5 samples are suffiscent
        
        # drive the separation distance of the two clusters to classify
        s=1/(d**(coef_s))
      
        fig, (axd,ax) = plt.subplots(1, 2, sharex=False,figsize=set_size(ratio=1),num=num)
        
        label=True
        RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
        y,X=RegObs.datas
        RegObs.plotOutputs(axd)
        axd.set_title("statistics of outputs $y_i=\\sigma(x_i^T \\theta)$")
        
        XP_HighDim_LogReg_Lattent(np.array([ax]),sigma0,mu0,N,d,list_p,c,ppca,svd,nbInnerLoop,seed,\
                                  label,coef_s,computeLaplace=True,loss=np.array(["kl"]),nbSamplesKL=nbSamplesKL)
        ax.set_title(' Logistic Regression'+'\n'+'$N={0}$, $d={1}$, $\sigma_0$={2}'.format(N,d,sigma0))  
        ax.set_yscale('log')
        ax.legend(loc="upper right", ncol=2)
        ax.set_xlabel('number of iterations')
        ax.set_ylabel('KL')
        ax.grid()
        
        plt.tight_layout()
        
        plt.savefig('./outputs/LRVGA_LogReg_p_{0}_{1}'.format(N,d))
        num=num+1
    
    ########################################################################################"
    #  EXPERIMENTS FOR GENERAL CASE (Section 5.4)
    ########################################################################################
    # Results of Section 5.4: 
    # 10 samples are suffiscent for mirror prox to converge 
    if 'Sampling' in Test:
        print("######### Sampling : Mirror prox with 1, 10 an 100 samples ##############")
        sigma0_list=[1,2,3]
        mu0=0
        d=1000
        p=10
        N=1000
        d_list=[d,d,d]
        p_list=[p,p,p]
        nbInnerLoop_list=[1,1,1]
        c=1
        seed=1
        ppca=False
        svd=False
        nbSamplesKL=4
        list_methods=["RVGA-extragrad","RVGA-sampled-extragrad-1","RVGA-sampled-extragrad-10","RVGA-sampled-extragrad-100"]
        list_labels=["baseline","1-sample","10-samples","100-samples"]
        list_cols=np.array(['k','r','b','g'])
        coef_s=0.15
        
        s=1/d**(coef_s)
        RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
        y,X=RegObs.datas
        fig, ax = plt.subplots(1, 1, sharex=False,figsize=set_size(ratio=1),num=num)
        RegObs.plotOutputs(ax)
        ax.set_title(r'statistics of outputs for the true lattent parameter $y_i=sigma(x_i^Ttheta)$')
        plt.savefig('./outputs/LRVGA_LogReg_vsp_Inputs_{0}_{1}'.format(N,d))
        num=num+1
        
        
        #fig, axs = plt.subplots(1, 3, sharex=False,figsize=set_size(ratio=0.75),num=num)
        fig, axs = plt.subplots(1, 3, sharex=False,figsize=(8, 4),num=num)
        
        XP_HighDim_LogReg_DataSET(axs,list_methods,list_labels,sigma0_list,mu0,N,d_list,p_list,c,ppca,svd,\
                              nbInnerLoop_list,seed,coef_s,computeLaplace=False,nbSamplesKL=nbSamplesKL,loss="kl",list_col0=list_cols)  
        fig.legend(loc='upper right',ncol=2,fontsize="small")
        fig.suptitle(r' Logistic Regression' +'\n' + ' sensitivity to number of samples'+'\n'+ '$d={0}$, $N={1}$, $p={2}$, $c={3}$'.format(d,N,p,c)) 
        fig.tight_layout()
        plt.savefig('./outputs/Sampling2_KL')
        num=num+1
        
    # Results of Appendix E 
    # Mirror prox (extragrad) is less biased than implicit or explicit scheme
    if 'MirrorProx' in Test:
        print("######### MirrorProx : Test of extragrad on a 2D example ##############")
        N=10
        d=2
        mu0=0
        c=1
        seed=1
        sigma0=10
        s=6
        
        RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
        y,X=RegObs.datas
        fig, ax = plt.subplots(1, 1, sharex=False,figsize=set_size(ratio=1),num=num)
        RegObs.plotOutputs(ax)
        num=num+1
        
        fig, axs = plt.subplots(1, 4, sharex=False,figsize=set_size(ratio=0.5),num=num)
        XP_2D(axs,sigma0,mu0,N,s,c,seed,nbLevels=1)
        fig.suptitle(r' Logistic Regression with '+r'$N={0}$, $\sigma_0$={1}, $s={2}$'.format(N,sigma0,s))    
        plt.legend(loc="lower center",ncol=3)
        plt.tight_layout()
        plt.savefig('./outputs/MirrorProx1_cov2D')
        num=num+1
    
    ########################################################################################"
    # EXTRA EXPERIMENTS (NOT IN THE PAPER)
    ########################################################################################
    if 'SamplingVsExtragrad' in Test:
        print("######### SamplingVsExtragrad : Test of sampling with/without extragrad ##############")
        N=1000
        d=100
        p=10
        sigma0_list=[1,5,10] #different values of prior sigma0 correspond to different hypothesis of smoothness for the posterior
        mu0=0
        d_list=[d,d,d]
        p_list=[p,p,p]
        nbInnerLoop_list=[1,1,1]
        c=1
        seed=1
        ppca=False
        svd=False
        nbSamplesKL=4
        list_labels=["RVGA","RVGA-sampled","RVGA-sampled-extragrad"]
        list_methods=["RVGA","RVGA-sampled-10","RVGA-sampled-extragrad-10"]
        list_cols=np.array(['b','r','g'])
        coef_s=0.15
        
        s=1/d**(coef_s)
        RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
        y,X=RegObs.datas
        fig, ax = plt.subplots(1, 1, sharex=False,figsize=set_size(ratio=0.5),num=num)
        RegObs.plotOutputs(ax)
        ax.set_title(r'statistics of outputs for the true lattent parameter $y_i=sigma(x_i^Ttheta)$')
        plt.savefig('./outputs/LRVGA_LogReg_vsp_Inputs_{0}_{1}'.format(N,d))
        num=num+1
        
        fig, axs = plt.subplots(1, 3, figsize=set_size(ratio=0.7),num=num)
        
        XP_HighDim_LogReg_DataSET(axs,list_methods,list_labels,sigma0_list,mu0,N,d_list,p_list,c,ppca,svd,\
                              nbInnerLoop_list,seed,coef_s,computeLaplace=False,nbSamplesKL=nbSamplesKL,loss="kl",list_col0=list_cols)  
        fig.suptitle(r' Logistic Regression: sensitivity to extra-grad'+'\n'+ '$d={0}$, $N={1}$, $p={2}$, $c={3}$'.format(d,N,p,c)) 
        fig.legend()
        plt.tight_layout()
        plt.savefig('./outputs/Sampling1_KL')
        num=num+1
        
    # the mirror prox is better than the implicit and explicit scheme 
    if 'MirrorProxHighDim' in Test:
        print("######### MirrorProxHighDim : Test of mirror prox without factor analysis ##############")
        sigma0_list=[1,3,10]
        mu0=0
        d=100
        p=10
        N=1000
        d_list=[d,d,d]
        p_list=[p,p,p]
        nbInnerLoop_list=[1,1,1]
        c=1
        seed=2
        ppca=False
        svd=False
        nbSamplesKL=5
        list_labels=["rvga-explicit","rvga-implicit","rvga-mirrorProx"]
        list_methods=["Full-RVGA-explicit","Full-RVGA","Full-RVGA-mirrorProx"]
        list_cols=np.array(['r','b','g'])
        coef_s=0.15
        
        s=1/d**(coef_s)
        RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
        y,X=RegObs.datas
        fig, ax = plt.subplots(1, 1, sharex=False,figsize=set_size(ratio=0.7),num=num)
        RegObs.plotOutputs(ax)
        ax.set_title(r'statistics of outputs for the true lattent parameter $y_i=sigma(x_i^Ttheta)$')
        plt.savefig('./outputs/LRVGA_LogReg_vsp_Inputs_{0}_{1}'.format(N,d))
        num=num+1
        
        fig, axs = plt.subplots(1, 3, sharex=False,figsize=set_size(ratio=0.7),num=num)
        
        XP_HighDim_LogReg_DataSET(axs,list_methods,list_labels,sigma0_list,mu0,N,d_list,p_list,c,ppca,svd,\
                              nbInnerLoop_list,seed,coef_s,computeLaplace=False,nbSamplesKL=nbSamplesKL,loss="kl",list_col0=list_cols)  
        fig.suptitle(r' Logistic Regression KL ' +'\n'+' extragrad vs implicit/explicit'+'\n'+ '$d={0}$, $N={1}$, $p={2}$, $c={3}$'.format(d,N,p,c)) 
        fig.legend()
        plt.tight_layout()
        plt.savefig('./outputs/MirrorProx2_KL')
        num=num+1
        
        
        fig, axs = plt.subplots(1, 3, sharex=False,figsize=set_size(ratio=0.7),num=num)
        
        XP_HighDim_LogReg_DataSET(axs,list_methods,list_labels,sigma0_list,mu0,N,d_list,p_list,c,ppca,svd,\
                              nbInnerLoop_list,seed,coef_s,computeLaplace=False,nbSamplesKL=nbSamplesKL,loss="distanceMAP",list_col0=list_cols)  
        fig.suptitle(r' Logistic Regression dist. to MAP'+'\n'+' extragrad vs implicit/explicit'+'\n'+ '$d={0}$, $N={1}$, $p={2}$, $c={3}$'.format(d,N,p,c)) 
        fig.legend()
        plt.tight_layout()
        plt.savefig('./outputs/MirrorProx2_MAP')
        num=num+1
    
    # the partial mirror prox is better than the full mirror prox when we use factor analysis
    if 'MirrorProxHighDimFA' in Test:
        print("######### MirrorProxHighDimFA : Test of mirror prox with factor analysis ##############")
        sigma0_list=[1,3,10]
        mu0=0
        d=100
        p=10
        N=1000
        d_list=[d,d,d]
        p_list=[p,p,p]
        nbInnerLoop_list=[1,1,1]
        c=1
        seed=2
        ppca=False
        svd=False
        nbSamplesKL=5
        list_labels=["mirrorProx-Full","mirrorProx-FA","mirrorProx-FA-sampled","s-mirrorProx-FA-sampled"]
        list_methods=["Full-RVGA-mirrorProx","RVGA-extragrad","RVGA-sampled-extragrad2-100","RVGA-sampled-extragrad-100"]
        list_cols=np.array(['k','r','b','g'])
        coef_s=0.15
        
        s=1/d**(coef_s)
        RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
        y,X=RegObs.datas
        fig, ax = plt.subplots(1, 1, sharex=False,figsize=set_size(ratio=0.7),num=num)
        RegObs.plotOutputs(ax)
        ax.set_title(r'statistics of outputs for the true lattent parameter $y_i=sigma(x_i^Ttheta)$')
        plt.savefig('./outputs/LRVGA_LogReg_vsp_Inputs_{0}_{1}'.format(N,d))
        num=num+1
        
       
        fig, axs = plt.subplots(1, 3, sharex=False,figsize=set_size(ratio=0.7),num=num)
        
        XP_HighDim_LogReg_DataSET(axs,list_methods,list_labels,sigma0_list,mu0,N,d_list,p_list,c,ppca,svd,\
                              nbInnerLoop_list,seed,coef_s,computeLaplace=False,nbSamplesKL=nbSamplesKL,loss="kl",list_col0=list_cols)  
        fig.suptitle(r' Logistic Regression: KL' +'\n'+'  sensitivity to extra-grad'+'\n'+ '$d={0}$, $N={1}$, $p={2}$, $c={3}$'.format(d,N,p,c)) 
        fig.legend()
        plt.tight_layout()
        plt.savefig('./outputs/MirrorProx3_KL')
        num=num+1
        
        
        fig, axs = plt.subplots(1, 3, sharex=False,figsize=set_size(ratio=0.7),num=num)
        
        XP_HighDim_LogReg_DataSET(axs,list_methods,list_labels,sigma0_list,mu0,N,d_list,p_list,c,ppca,svd,\
                              nbInnerLoop_list,seed,coef_s,computeLaplace=False,nbSamplesKL=nbSamplesKL,loss="distanceMAP",list_col0=list_cols)  
        fig.suptitle(r' Logistic Regression: dist. to MAP' +'\n'+' sensitivity to extra-grad'+'\n'+ '$d={0}$, $N={1}$, $p={2}$, $c={3}$'.format(d,N,p,c)) 
        fig.legend()
        plt.tight_layout()
        plt.savefig('./outputs/MirrorProx3_MAP')
        num=num+1
    
    
    
        
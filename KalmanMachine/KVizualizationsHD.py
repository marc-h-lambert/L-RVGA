###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Plotting routines to benchmark different version of the LRVGA algorithm         #
###################################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.linalg as LA

# plot the covariances resulting from several algorithms stored in list_kf
def plotCov(axs,list_kf,list_labels,posterior,nbLevels=1):
    
    nbFig= list_kf.shape[0]
    assert(nbFig==axs.shape[0])
    assert(nbFig==list_labels.shape[0])
    i=0
    labelLap=True
    for ax in axs:
            
        list_kf[i].plotEllipsoid(ax,nbLevels=nbLevels,labelize=labelLap)
        if not posterior is None:
            posterior.plot(ax,labelize=labelLap,showMleMap=True)
        ax.set_title(list_labels[i])
        ax.set_aspect('equal')
        labelLap=False
        i=i+1
            
# plot the KL divergence in Linear regression from several algorithms stored in list_lskf  
def plotLSKLlinReg_lattent(ax,LKF,list_lskf,list_p,truePosterior,labelize=True):    
    N,d=LKF.history_theta.shape
    
    tic = time.perf_counter()
    kl_histo_LKF=truePosterior.onlineKL(LKF)[:-2]
    toc = time.perf_counter()
    print('Compute KL for LKF ... in {0:.2}'.format(toc-tic))
    if labelize:
        ax.plot(np.arange(0,kl_histo_LKF.shape[0]),kl_histo_LKF,label='Full KF')
                
    for i in range(0,len(list_lskf)):
        p=list_p[i]
        lskf=list_lskf[i]
        tic = time.perf_counter()
        kl_histo=truePosterior.onlineKL_LargeScale(lskf)[:-2]
        toc = time.perf_counter()
        print('Compute KL for LSKF-EM (p={0})... in {1:.2}'.format(p,toc-tic))
    
        if labelize:
            ax.plot(np.arange(0,kl_histo.shape[0]),\
                    kl_histo,label='p={}'.format(p))
        else:
            ax.plot(np.arange(0,kl_histo.shape[0]),\
                    kl_histo)
        
                
# plot the KL divergence in Logistic regression from several algorithms stored in list_lskf  
def plotLSKLlogReg(ax,list_lskf,list_methods,list_labels,truePosterior,labelize=True,seed=10,\
                   computeLaplace=True,nbSamplesKL=10,largeScale=True,list_col=[],list_linestyle=[]):    
    N,d=list_lskf[0].history_theta.shape
    
    np.random.seed(seed)
    normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamplesKL,))
    if computeLaplace:
        kl_lap=truePosterior.KL_Laplace(normalSamples)
    for i in range(0,len(list_lskf)):
        label=list_labels[i]
        lskf=list_lskf[i]
        method=list_methods[i]
        if len(list_col)!=0:
            col=list_col[i]
        if len(list_linestyle)==0:
            linestyle='-'
        else:
            linestyle=list_linestyle[i]
        tic = time.perf_counter()
        if not largeScale or "Full" in method:
            kl_histo=truePosterior.onlineKL(lskf,nbSamplesKL,seed)[:-4]
        else:
            kl_histo=truePosterior.onlineKL_LargeScale(lskf,nbSamplesKL,seed)[:-4]
        toc = time.perf_counter()
        print('KL for {0} = {1} (computed in {2:.2})'.format(label,kl_histo[-1],toc-tic))
            
        idx0=0
        if labelize:
            if len(list_col)!=0:
                ax.plot(np.arange(0,kl_histo[idx0:].shape[0]),\
                            kl_histo[idx0:],label=label,color=col,linestyle=linestyle,linewidth='1.5')
            else:
                ax.plot(np.arange(0,kl_histo[idx0:].shape[0]),\
                            kl_histo[idx0:],label=label,linestyle=linestyle,linewidth='1.5')
        else:
            if len(list_col)!=0:
                ax.plot(np.arange(0,kl_histo[idx0:].shape[0]),\
                            kl_histo[idx0:],color=col,linewidth='1.5')
            else:
                ax.plot(np.arange(0,kl_histo[idx0:].shape[0]),\
                            kl_histo[idx0:],color=col,linewidth='1.5')
                
    if computeLaplace:
        if labelize:
            ax.plot(np.arange(0,kl_histo[idx0:].shape[0]),\
                        np.ones(kl_histo[idx0:].shape[0],)*kl_lap,label='Laplace',color='k',linestyle='dashed',linewidth='1')
        else:
            ax.plot(np.arange(0,kl_histo[idx0:].shape[0]),\
                    np.ones(kl_histo[idx0:].shape[0],)*kl_lap,color='k',linestyle='dashed',linewidth='1')
    

# plot the distance from MAP in Logistic regression from several algorithms stored in list_lskf   
def plotLSerrorMaplogReg(ax,list_lskf,list_labels,truePosterior,labelize=True,list_col=[]):    
    N,d=list_lskf[0].history_theta.shape
    
    for i in range(0,len(list_lskf)):
        label=list_labels[i]
        lskf=list_lskf[i]
        if len(list_col)==0:
            col='k'
        else:
            col=list_col[i]
        map_histo=truePosterior.onlineDistancefromMap(lskf)
        if labelize:
            ax.plot(np.arange(0,map_histo.shape[0]),\
                    map_histo,label=label,color=col)
        else:
            ax.plot(np.arange(0,map_histo.shape[0]),\
                    map_histo,color=col) 
    ax.locator_params(axis="x", nbins=5)
    ax.set_xlabel('number of iterations')
    ax.locator_params(axis="y", nbins=5)
    ax.set_ylabel(r'$||\mu_t-\mu_{map}||$')

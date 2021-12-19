###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Define metrics to assess linear and logistic regression    :                    #
# - the posterior including the Laplace approximation                             #                                    #
# - the KL divergence                                                             #                                                                 #
###################################################################################

import numpy as np 
import numpy.linalg as LA
import math
from .KUtils import graphix, bayesianlogisticPdf,negbayesianlogisticPdf,\
    neglogisticPdf, importanceSamples, fastLogDet, FAInverse, bayesianlogisticPdf_largeScale
from .Kalman4LogisticReg import LaplaceLogisticRegression
from scipy import optimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time, tracemalloc
    
# Bayesian Logistic Regression (with a Gaussian model)
class TruePosterior(object):
    
    def __init__(self,theta0):
        super().__init__()
        self._theta0=theta0 # the initial guess
        self._X=None
        self._Y=None
        self._Cov=None
        self._map=None
        self._mle=None
        self._xv=None
        self._yv=None
        self._pdf=None
        self._lossAtMap=None
     
    def fit(self,X,Y,sigma=1,Nsamples=30):        
        pass
    
    def plot(self,ax,labelize=True,showMap=True,showMle=False,n_contours=4):
        if not (self._pdf is None):
            CS=ax.contour(self._xv,self._yv,self._pdf,n_contours,zorder=1,\
                          extent=(self._xv[0,0],self._xv[0,-1],self._yv[0,0],self._yv[-1,0]))

            if showMap: 
                if labelize:
                    ax.scatter(self._map[0],self._map[1],marker='o',color='k',label='MAP'.format(self._lossAtMap),linewidth=1)
                else:
                    ax.scatter(self._map[0],self._map[1],marker='o',color='k')
            if showMle: 
                if labelize:
                    ax.scatter(self._mle[0],self._mle[1],marker='o',color='g',label='MLE',linewidth=1)
                else:
                    ax.scatter(self._mle[0],self._mle[1],marker='o',color='g')
                    
                    
############## METRICS FOR LINEAR REGRESSION ##########################        
class PosteriorLinReg(TruePosterior):
    
    def __init__(self,theta0, Cov0):
        super().__init__(theta0)
        self._Cov0=Cov0 # the initial covariance (ie uncertainty on the initial guess)
        
    def fit(self,X,Y,sigma=1,Nsamples=30):
        self._X=X
        N,d=X.shape
        Y=Y.reshape(N,1)
        self._Y=Y
        
        invCov0=LA.inv(self._Cov0)
        self._Cov=LA.inv(invCov0+X.T.dot(X)/sigma**2)
        self._map=self._Cov.dot(X.T.dot(Y)/sigma**2+invCov0.dot(self._theta0))
        self._mle=LA.inv(X.T.dot(X)/sigma**2).dot(X.T.dot(Y)/sigma**2)
        self._invCov=LA.inv(self._Cov)
        (sign, logdet) = LA.slogdet(self._Cov)
        self._logDetCov=logdet
        self._lossAtMap=0#-0.5*math.log(math.pow(2*math.pi,d))-0.5*self._logDetCov
        
        if d==2:    
            # generate the true posterior for different values of theta
            std=np.max(np.sqrt(np.linalg.eigvals(self._Cov))).item()*3
            theta1=np.linspace(self._map[0]-std,self._map[0]+std,Nsamples)
            theta2=np.linspace(self._map[1]-std,self._map[1]+std,Nsamples)
            self._pdf=np.zeros((Nsamples,Nsamples))    
            self._xv,self._yv=np.meshgrid(theta1,theta2)
            for i in np.arange(0,Nsamples):
                for j in np.arange(0,Nsamples):
                    theta=np.zeros((2,1))
                    theta[0]=self._xv[i,j]
                    theta[1]=self._yv[i,j]
                    self._pdf[i,j]=multivariate_normal.pdf(theta.reshape(d,),self._map.reshape(d,),self._Cov)
        else:
            print('Can not draw true posterior grid for d != 2.')
                
        return self
        
    # divergence of the true posterior relative to an approximated Gaussian distribution
    # (exact  analytical)  
    def divergence(self,mu,Cov):
        d=mu.shape[0]
        (sign, logdetCov) = LA.slogdet(Cov)
        Result=self._logDetCov-logdetCov-d
        Result+=np.trace(self._invCov.dot(Cov))
        Result+=(self._map-mu).T.dot(self._invCov).dot(self._map-mu)
        return 0.5*Result
    
    def divergenceLargeScale(self,mut,Bt,psit):
        d=mut.shape[0]
        logdetCovt=fastLogDet(psit,Bt)
        Covt=FAInverse(psit,Bt)        
        Result=self._logDetCov+logdetCovt-d
        Result+=np.trace(self._invCov.dot(Covt))
        Result+=(self._map-mut).T.dot(self._invCov).dot(self._map-mut)
        return 0.5*Result
    
    def onlineKL(self,onlineBayes):
        N,d=onlineBayes.history_theta.shape
        history_kl=np.zeros([N,1])
        for t in range(0,N):
            thetat=onlineBayes.history_theta[t].reshape(d,1)
            Covt=onlineBayes.history_Cov[t].reshape(d,d)
            history_kl[t]=self.divergence(thetat,Covt)
        return history_kl  
    
    def onlineKL_LargeScale(self,onlineBayes):
        N,d=onlineBayes.history_theta.shape
        p=onlineBayes.dimLattent
        history_kl=np.zeros([N,1])
        for t in range(0,N):
            thetat=onlineBayes.history_theta[t].reshape(d,1)
            Bt=onlineBayes.history_B[t].reshape(d,p)
            psit=onlineBayes.history_psi[t].reshape(d,1)
            history_kl[t]=self.divergenceLargeScale(thetat,Bt,psit)
        return history_kl 
    
############## METRICS FOR LOGISTIC REGRESSION ##########################    

class PosteriorLogReg(TruePosterior):
    
    def __init__(self,theta0, psi0, W0, computeLaplace=True):
        super().__init__(theta0)
        d,p=W0.shape
        self._invCov0=np.diag(psi0.reshape(d,))+W0.dot(W0.T)
        self._Cov0=FAInverse(psi0,W0)
        self._logdetCov0=fastLogDet(psi0,W0)
        self._psi0=psi0
        self._W0=W0
        self._lapCov=None
        self.memoryUsedMLE=-1
        self.timeCostMLE=-1
        self.memoryUsedLAP=-1
        self.timeCostLAP=-1
        self.computeLaplace=computeLaplace
        self._map=None
        self._lossAtMap=0
        
    # compute the posterior (un-normalized by default)
    def logPdf(self,theta):
        return bayesianlogisticPdf(theta,self._theta0, self._invCov0,self._logdetCov0,self._X,self._Y,Z=1)
    
    # compute the posterior (un-normalized by default)
    def logPdf_largeScale(self,theta):
        return bayesianlogisticPdf_largeScale(theta,self._theta0, self._W0, self._psi0, self._logdetCov0, self._X,self._Y,Z=1)
        
    def fit(self,X,Y,sigma=1,Nsamples=30,monitor=False):
        self._X=X
        self._Y=Y
        N,d=X.shape
        
        # solve with Laplace to find the center of the grid
        # Compute the batch MAP (with Laplace) theta0 suppposed null and Cov0 diagonal !!
        if self.computeLaplace:            
            self.lap = LaplaceLogisticRegression(self._theta0,self._Cov0).fit(X, Y.reshape(N,))
            self._map= self.lap.maxP
            self._lapCov=self.lap.Cov
        else:
            logdetCov0=logdetCov
            invCov0=LA.inv(self._Cov0)
            sol=optimize.minimize(negbayesianlogisticPdf, self._theta0, args=(self._theta0,self._invCov0,self._logdetCov0,X,y.reshape(N,),1,),method='L-BFGS-B')
            self._map=sol.x
        
        # Compute MLE
        sol_mle=optimize.minimize(neglogisticPdf, self._theta0, args=(X,Y.reshape(N,),1,),method='L-BFGS-B')
        self._mle=sol_mle.x
            
        if d==2:
            std=np.max(np.sqrt(np.linalg.eigvals(self.lap.Cov))).item()*2
    
            # generate the true posterior for different values of theta
            theta1=np.linspace(self.lap.maxP[0]-std,self.lap.maxP[0]+std,Nsamples)
            theta2=np.linspace(self.lap.maxP[1]-std,self.lap.maxP[1]+std,Nsamples)
            self._pdf=np.zeros((Nsamples,Nsamples))    
            self._xv,self._yv=np.meshgrid(theta1,theta2)
            for i in np.arange(0,Nsamples):
                for j in np.arange(0,Nsamples):
                    theta=np.zeros((2,1))
                    theta[0]=self._xv[i,j]
                    theta[1]=self._yv[i,j]
                    self._pdf[i,j]=np.exp(self.logPdf(theta))
            self._lossAtMap=self.logPdf(np.array([self.lap.maxP[0],self.lap.maxP[1]]))[0][0]
                
        return self
    
    def plot(self,ax,labelize=True,showMleMap=True):
        super().plot(ax,labelize,showMleMap)
        if showMleMap: 
            if labelize:
                graphix.plot_ellipsoid2d(ax,self._map,self.lap.Cov,col='k',linewidth=1,zorder=3,linestyle='--',label='Laplace')
            else:
                graphix.plot_ellipsoid2d(ax,self._map,self.lap.Cov,col='k',linewidth=1,zorder=3,linestyle='--')

    def KL_Laplace(self,normalSamples):
        (sign, logdetCov0) = LA.slogdet(self._Cov0)
        return self.divergence(self.lap.maxP,self.lap.Cov,normalSamples)
    
    # divergence of the true posterior relative to an approximated Gaussian distribution
    # (approximated with MC sampling) 
    # we suppose we have already samples following N(O,I) in "normalSamples" 
    def divergence(self,theta,Cov,normalSamples):
        d=theta.shape[0]
        theta=theta.reshape(d,1)
        (sign, logdet) = LA.slogdet(Cov)
        entropy=0.5*logdet+d/2*(1+math.log(2*math.pi))
        A=0
        
        thetaVec=theta+np.linalg.cholesky(Cov).dot(normalSamples.T)
        cmpt=0
        nbSamplesKL=normalSamples.shape[0]
        for i in range(0,nbSamplesKL):
            thetai=thetaVec[:,i].reshape(d,1)
            A=A-self.logPdf(thetai)
        KL=A/nbSamplesKL-entropy
        return KL.item()
    
    # divergence of the true posterior relative to an approximated Gaussian distribution
    # (approximated with MC sampling)-->  use here the factor analysis form Cov=inv(BB+psi)
    def divergenceLargeScale(self,mu,B,psi,normalSamples):
        d=mu.shape[0]
        mu=mu.reshape(d,1)
        nbSamplesKL=normalSamples.shape[0]

        logdet=fastLogDet(psi,B)
        entropy=-0.5*logdet+d/2*(1+math.log(2*math.pi))
        
        # importanceSampling may be used in high dim 
        # if cholesky fail but more approximated
        importanceSampling=False
        if importanceSampling: 
            thetaVec,pVec=importanceSamples(mu,B,psi,normalSamples)
            thetaVec=thetaVec.T
        else:
            Cov=FAInverse(psi,B)
            thetaVec=mu+np.linalg.cholesky(Cov).dot(normalSamples.T)
            pVec=np.ones([nbSamplesKL,1])/nbSamplesKL
        
        cmpt=0
        A=0
        for i in range(0,nbSamplesKL):
            thetai=thetaVec[:,i].reshape(d,1)
            pi=pVec[i]
            A=A-self.logPdf_largeScale(thetai)*pi
        KL=A-entropy
        return KL.item()
    
    def onlineKL(self,onlineBayes,nbSamplesKL,seed):
        N,d=onlineBayes.history_theta.shape
        history_kl=np.zeros([N,1])
        np.random.seed(seed)
        normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamplesKL,))
        for t in range(0,N):
            thetat=onlineBayes.history_theta[t].reshape(d,1)
            Covt=onlineBayes.history_Cov[t].reshape(d,d)
            history_kl[t]=self.divergence(thetat,Covt,normalSamples)
        self.divergence(thetat,Covt,normalSamples)
        return history_kl  
    
    def onlineKL_LargeScale(self,onlineBayes,nbSamplesKL,seed):
        N,d=onlineBayes.history_theta.shape
        p=onlineBayes.dimLattent
        history_kl=np.zeros([N,1])
        np.random.seed(seed)
        normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamplesKL,))
        for t in range(0,N):
            thetat=onlineBayes.history_theta[t].reshape(d,1)
            Bt=onlineBayes.history_B[t].reshape(d,p)
            psit=onlineBayes.history_psi[t].reshape(d,1)
            history_kl[t]=self.divergenceLargeScale(thetat,Bt,psit,normalSamples)
        return history_kl  
    
    def onlineDistancefromMap(self,onlineBayes):
        N,d=onlineBayes.history_theta.shape
        history_error=np.zeros([N,1])
        for t in range(0,N):
            thetat=onlineBayes.history_theta[t].reshape(d,1)
            history_error[t]=LA.norm(self._map.reshape(d,1)-thetat)
        return history_error                  
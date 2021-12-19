###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Bayesian regression based on the Gaussian approximation of posterior            #
# BayesianReg:  the general API which work for linear or logistic reg             #                                      #                               #
# OnlineBayesianRegression : the API for sequential algorithms (Kalman)           #
# LargeScaleBayesianRegression : the API for the factorized covariance version    #
# (A sikit learn like API is used but we support only the binary case and no bias #
#   ie classes_={0,1} and intercept_=0 )                                          #
###################################################################################

import numpy as np
import numpy.random
from .KUtils import graphix, FAInverse
from matplotlib import cm
from .KFactorAnalysis import CovarianceFactorAnalysisEM, CovarianceFactorAnalysisSVD
import tracemalloc, time

# Bayesian Logistic Regression (with a Gaussian model)
class BayesianRegression(object):
    
    def __init__(self,theta0):
        super().__init__()
        self._theta0=theta0 # the initial guess
        self._theta=np.copy(self._theta0) # the current mean
     
    def fit(self,X,y):        
        pass
    
    @property
    def theta(self):
        return self._theta
    
    @property
    def Cov(self):
        pass

import random
# Stochastic version of Bayesian Logistic Regression (with a Gaussian model)
class OnlineBayesianRegression(BayesianRegression):
    
    def __init__(self, theta0, Cov0, sigma, passNumber):
        super().__init__(theta0)
        self._Cov0=Cov0 # the initial covariance (ie uncertainty on the initial guess)
        self._passNumber=passNumber # the number of pass on datas
        self._history_theta=None # the mean history
        self._history_Cov=None  # the covariance history
        self._sigma = sigma # the noise on observation
        self._Cov=np.copy(Cov0)
        self.memoryUsed=-1
        self.timePerIteration=-1
    
    # virtual method
    def update(self,xt,yt):
        pass
        
    def fit(self,X,y,monitor=True):
        N,d=X.shape
        self._history_theta = np.zeros((N*self._passNumber+1,d))
        self._history_theta[0,:]=self._theta0.flatten()
        self._history_Cov = np.zeros((N*self._passNumber+1,d*d))
        self._history_Cov[0,:]=self._Cov0.flatten()
        nbIter=1
        #print('self._passNumber',self._passNumber)
        for numeroPass in range(1,self._passNumber+1):   
            for t in range(0,N): 
                # get new observation
                yt=y[t].reshape(1,1)
                xt=X[t].reshape(d,1)
                
                if monitor and t==0:
                    tracemalloc.start()
                    tic=time.perf_counter()
                    
                self.update(xt,yt)
                
                if monitor and t==0:
                    toc=time.perf_counter()
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    self.memoryUsed=current/10**6
                    self.timePerIteration=toc-tic
                
                self._history_theta[nbIter,:]=self._theta.flatten()
                self._history_Cov[nbIter,:]=self._Cov.flatten()
            
                nbIter=nbIter+1
        
            if numeroPass>1:
                # To manage different pass, shuffle the dataset
                DataSet=list(zip(X,y))
                random.shuffle(DataSet)
                X,y = zip(*DataSet)
        #print(nbIter)
        return self
        
    @property
    def history_theta(self):
        return self._history_theta
    
    @property
    def history_Cov(self):
        return self._history_Cov
    
    @property
    def Cov(self):
        return self._Cov
    
    def _cov(self,i):
        d=self._theta0.shape[0]
        if not self._history_Cov is None :
            return self._history_Cov[i].reshape(d,d)
        return None
    
    # u, v: coordinates of projection if projOnCoords=True
    # else they are vectors
    @staticmethod
    def PlotEllipsoids(_history_theta,_cov,ax,u=0,v=1,nbLevels=6,labelize=True,projOnCoords=True,idx0=1):
                
        colorMap = cm.get_cmap('bwr', 256)
        colorCovs = colorMap(np.linspace(0, 1, nbLevels))

        d=_history_theta[0].shape[0]
        N=_history_theta.shape[0]-1
        lw=1.6
        
        # Print final ellipsoid
        if projOnCoords:
            thetaproj,Covproj=graphix.projEllipsoid(_history_theta[-1],_cov(-1),u,v)
        else:
            thetaproj,Covproj=graphix.projEllipsoidOnVector(_history_theta[-1],_cov(-1),u,v)
        
        if labelize:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,'r',linewidth=lw,zorder=3,linestyle='-',label='last iteration')
        else:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,'r',linewidth=lw,zorder=3,linestyle='-')
    
        if nbLevels>1:
            # Print intermediate ellipsoids 
            l=int(N/(nbLevels))
            idx=idx0   
            col=0
            for i in range(0,nbLevels):
                if projOnCoords:
                    thetaproj,Covproj=graphix.projEllipsoid(_history_theta[idx],_cov(idx),u,v)
                else:
                    thetaproj,Covproj=graphix.projEllipsoidOnVector(_history_theta[idx],_cov(idx),u,v)
                
                if idx==idx0 and labelize:
                    graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,colorCovs[col],linewidth=lw,zorder=3,label='first iteration',linestyle='--')
                else:
                    graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,colorCovs[col],linewidth=lw)
                idx=min(idx+l,N)
                col=col+1
            
    def plotEllipsoid(self,ax,u=0,v=1,nbLevels=6,labelize=True,projOnCoords=True,idx0=0):
        return OnlineBayesianRegression.PlotEllipsoids(self._history_theta,self._cov,ax,u,v,\
                                                       nbLevels,labelize,projOnCoords,idx0)

from math import sqrt
# Stochastic version of Bayesian Logistic Regression (with a Gaussian model)
class LargeScaleBayesianRegression(BayesianRegression):
    
    def __init__(self, theta0, psi0, B0, passNumber=1, sigma=1, ppca=False, svd=False,nbInnerLoop=50):
        super().__init__(theta0)
        self._history_theta=None # the mean history
        self._history_B=None # the partial cov history
        self._history_psi=None # the partial cov history
        self._passNumber=passNumber # the number of pass on datas
        self._svd=svd
        self._psi0=psi0
        self._B0=B0
        self._sigma = sigma # the noise on observation
        if svd:
            self._covAnalyze = CovarianceFactorAnalysisSVD(psi0, B0, ppca=ppca)
        else:
            self._covAnalyze = CovarianceFactorAnalysisEM(psi0, B0, ppca=ppca,nbInnerLoop=nbInnerLoop)
        self.memoryUsed=-1
        self.timePerIteration=-1
        
    # virtual method
    def update(self,xt,yt):
        pass
        
    # fit one observation, if monitor = true, we print the memory and time cost 
    def fit(self,X,y,monitor=True):
        N,d=X.shape
        p=self._B0.shape[1]
        if monitor:
            self._history_theta = np.zeros((N*self._passNumber+1,d))
            self._history_theta[0,:]=self._theta0.flatten()
            self._history_B = np.zeros((N*self._passNumber+1,d*p))
            self._history_psi=np.zeros((N*self._passNumber+1,d))
            self._history_B[0,:]=self._B0.flatten()
            self._history_psi[0,:]=self._psi0.flatten()
        nbIter=1
        for numeroPass in range(1,self._passNumber+1):   
            for t in range(0,N): 
                
                # get new observation
                yt=y[t].reshape(1,1)
                xt=X[t].reshape(d,1)
                
                # update theta and Cov using the new observation
                if monitor and t==0:
                    tracemalloc.start()
                    tic=time.perf_counter()
                self.update(xt,yt)
                
                if monitor and t==0:
                    toc=time.perf_counter()
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    self.memoryUsed=current/10**6
                    self.timePerIteration=toc-tic
                
                if monitor:
                    self._history_theta[nbIter,:]=self._theta.flatten()
                    self._history_B[nbIter,:]=self._covAnalyze._Wo.flatten()
                    self._history_psi[nbIter,:]=self._covAnalyze._psio.flatten()

                nbIter=nbIter+1
        
            if numeroPass>1:
                # To manage different pass, shuffle the dataset
                DataSet=list(zip(X,y))
                random.shuffle(DataSet)
                X,y = zip(*DataSet)
                
        return self
    
    @property
    def psi(self):
        return self._covAnalyze._psio
    
    @property
    def history_psi(self):
        return self._history_psi
    
    @property
    def B(self):
        return self._covAnalyze._Wo
    
    @property
    def history_B(self):
        return self._history_B
    
    @property
    def dimLattent(self):
        return self._B0.shape[1]
        
    def _fisherMatrix(self,t):
        if t==0:
            return self._B0.dot(self._B0.T)+np.diag(self._psi0.reshape(-1,))
        if not self._history_B is None and not self._history_psi is None:
            d,p=self._B0.shape
            B=self._history_B[t].reshape(d,p)
            psi=self._history_psi[t].reshape(d,1)
            return (B.dot(B.T)+np.diag(psi.reshape(d,)))/t
        return None
    
    @property
    def history_theta(self):
        return self._history_theta
    
    @property
    def history_Cov(self):
        N=self._history_B.shape[0]
        d,p=self._B0.shape
        history_Cov = np.zeros((N*self._passNumber+1,d*d))
        for t in range(0,N): 
             history_Cov[t]=self._cov(t).flatten()
        return history_Cov
    
    # Cov in form (BBT+Psi)^-1
    @property
    def Cov(self):
        return FAInverse(self.psi,self.B)
    
    @property
    def FisherMatrix(self):
        return  (self.B.dot(self.B.T)+np.diag(self.psi.reshape(-1,)))/self._history_B.shape[0]
    
    def _cov(self,i):
        if not self._history_B is None and not self._history_psi is None:
            d,p=self._B0.shape
            B=self._history_B[i].reshape(d,p)
            psi=self._history_psi[i].reshape(d,1)
            return FAInverse(psi,B)
        return None
            
    def plotEllipsoid(self,ax,u=0,v=1,nbLevels=6,labelize=True,projOnCoords=True,idx0=1):
        return OnlineBayesianRegression.PlotEllipsoids(self._history_theta,self._cov,ax,u,v,\
                                                       nbLevels,labelize,projOnCoords,idx0)



    
###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Online second order method for linear regression (linear Kalman filter)         #                                                                                                              
###################################################################################

from .KUtils import sigmoid, sigp, sigpp
import numpy as np
import numpy.random
import numpy.linalg as LA
from .KBayesianReg import OnlineBayesianRegression, LargeScaleBayesianRegression
import math
from math import log, exp
from scipy import optimize

class LinearPredictor(object):
    
    def __init__(self):
        super().__init__()
        
    def predict(self,X):
        return X.dot(self.theta)
    
    #prediction of N outputs for inputs X=(N,d)
    def predict_proba(self,X):
        return np.diag(X.dot(self.Cov).dot(X.T))
    
    def plotPredictionMap(self,ax,size=6):
      N=100
      x=np.zeros([2,1])
      theta1=np.linspace(-size/2,size/2,N)
      theta2=np.linspace(-size/2,size/2,N)
      probaOutput=np.zeros((N,N)) 
      xv,yv=np.meshgrid(theta1,theta2)
      for i in np.arange(0,N):
          for j in np.arange(0,N):
              x[0]=xv[i,j]
              x[1]=yv[i,j]
              probaOutput[i,j]=self.predict_proba(x.T)
      contr=ax.contourf(xv,yv,probaOutput,20,zorder=1,cmap='jet')
      ax.set_xlim(-size/2, size/2)
      ax.set_ylim(-size/2, size/2)
      return contr
  
class LKFLinReg(OnlineBayesianRegression, LinearPredictor):
        
    def update(self,xt,yt):
        # intermediate variables
        d=xt.shape[0]
        nu=xt.T.dot(self._Cov.dot(xt))
        Pu=self._Cov.dot(xt)
            
        # update state
        self._Cov=self._Cov-Pu.dot(Pu.T)/(self._sigma**2+nu)
        K=self._Cov.dot(xt)/self._sigma**2
        err=yt-xt.T.dot(self._theta)
        self._theta=self._theta+K*err
        
from math import sqrt
class LargeScaleLKFLinReg(LargeScaleBayesianRegression, LinearPredictor):
     
    def update(self,xt,yt):   
        
        psi,B=self._covAnalyze.fit(xt,self._sigma)        
        
        # update state
        error=yt-xt.T.dot(self._theta)
                
        if psi.all()==0:
            invBB=LA.inv(B.dot(B.T))
            self._theta=self._theta+invBB.dot(xt)*error
        elif psi.any()==0:  
            print('Attention: self._psi.any()==0')
            invBB=LA.inv(B.dot(B.T))
            self._theta=self._theta+invBB.dot(xt)*error
        else:
            p=B.shape[1]
            invM=LA.inv(np.identity(p)+B.T.dot(B/psi))
            U=B.T.dot(xt/psi)
            self._theta=self._theta+((xt-B.dot(invM).dot(U))*error/self._sigma**2)/psi

    

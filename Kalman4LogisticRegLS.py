####################################################################################
# THE KALMAN MACHINE LIBRARY - LARGE SCALE (LS)                                    #
# Code supported by Marc Lambert                                                   #
####################################################################################
# Online second order method for Large Scale (LS) logistic regression :            #                #
# LargeScaleEKFLogReg :                                                            #
# The extended Kalman filter without averaging with FA factorization of P          # 
# LargeScaleRVGALogReg:                                                            #
# The implicit RVGA with analytical averaging with FA factorization of P           #
# LargeScaleRVGALogRegExplicit:                                                    #
# The explicit RVGA with analytical averaging with FA factorization of P           #
# LargeScaleRVGALogRegSampled:                                                     # 
# The explicit RVGA with sampling-averaging with FA factorization of P             #                                  #                                                                    #
# --> see "The limited memory recursive variational Gaussian approximation (L-RVGA)#              #
#     Marc Lambert, Silvere Bonnabel and Francis Bach 2020"                        #                                                                                                                  
####################################################################################

from .KUtils import sigmoid, sigp, FAInverse, importanceSamples, ensembleSamples
import numpy as np
import numpy.random
import numpy.linalg as LA
from .KBayesianReg import LargeScaleBayesianRegression
from .Kalman4LogisticReg import LogisticPredictor, RVGALogReg
import math

#  The natural gradient or EKF + factor analysis
class LargeScaleEKFLogReg(LargeScaleBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, psi0, B0, passNumber=1, ppca=False, svd=False,nbInnerLoop=50):
        super().__init__(theta0, psi0, B0, passNumber=passNumber, sigma=1, ppca=ppca, svd=svd,\
                         nbInnerLoop=nbInnerLoop)
    
    def update(self,xt,yt):
        # intermediate variables            
        a=xt.T.dot(self._theta)
        m=max(sigp(a),1e-100)
        sigma=np.sqrt(1/m)
        
        # update state
        self._covAnalyze.fit(xt,sigma) 
        self._theta=self._theta+self.Cov.dot(xt)*(yt-sigmoid(a))            

#  The implicit RVGA (with a Newton solver) + factor analysis
class LargeScaleRVGALogReg(LargeScaleBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, psi0, B0, passNumber=1, ppca=False, svd=False,nbInnerLoop=50):
        super().__init__(theta0, psi0, B0, passNumber=passNumber, sigma=1, ppca=ppca, svd=svd,nbInnerLoop=nbInnerLoop)
    
    def update(self,xt,yt):
        
        # (1) Compute the implicite scalar intermediate variables with Newton optim
        
        if self.psi.all()==0 or self.psi.any()==0: 
            # compute the inverse of the full matrix
            nu0=xt.T.dot(self.Cov.dot(xt))
        else:
            # use Woodbury formula with old values of B, psi
            p=self.B.shape[1]
            invM=LA.inv(np.identity(p)+self.B.T.dot(self.B/self.psi))
            U=self.B.T.dot(xt/self.psi)
            U2=(xt-self.B.dot(invM).dot(U))/self.psi  
            nu0=xt.T.dot(U2)
            
        alpha0=xt.T.dot(self._theta)
            
        a,nu,nbInnerLoop=RVGALogReg.optim2D(np.asscalar(alpha0), np.asscalar(nu0), np.asscalar(yt))
            
        # (2): Update first the STATE in the implicite scheme
        k=RVGALogReg.beta/math.sqrt(nu+RVGALogReg.beta**2)
        # compute the error
        error=yt-sigmoid(k*a)
        # Update the state
        if self.psi.all()==0 or self.psi.any()==0:
            self._theta=self._theta+self.Cov.dot(xt)*error
        else:
            # use the new updates
            #p=self.B.shape[1]
            #invM=LA.inv(np.identity(p)+self.B.T.dot(self.B/self.psi))
            #U=self.B.T.dot(xt/self.psi)
            self._theta=self._theta+error*(xt-self.B.dot(invM).dot(U))/self.psi 
        
        # (3): Update the COVARIANCE using Factor analysis
            
        # compute sigma to have 
        # the update in a factorized form invP=invP+xx^T/sigma
        m=max(k*sigp(k*a),1e-100)
        sigma=np.sqrt(1/m)
        # Do factor analysis and update the covariance matrix 
        # (self.B and self.psi are updates)
        self._covAnalyze.fit(xt,sigma) 

# The explicit RVGA (without Newton solver) + factor analysis
# The average is quasi exact (no sampling approximation) 
# If extragrad=True we update two times the mean 
# If extragrad=True and updateCovTwoTimes=True we update two times the covariance (mirror prox)
class LargeScaleRVGALogRegExplicit(LargeScaleBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, psi0, B0, passNumber=1, ppca=False, svd=False,nbInnerLoop=50,extragrad=True,updateCovTwoTimes=False):
        super().__init__(theta0, psi0, B0, passNumber=passNumber, sigma=1, ppca=ppca, svd=svd,\
                         nbInnerLoop=nbInnerLoop)
        self.extragrad=extragrad
        self.updateCovTwoTimes=updateCovTwoTimes
    
    @staticmethod
    def updateExpectationExact(xt,theta,B,psi):
        # compute the scalar intermediate variables
        
        if psi.all()==0 or psi.any()==0: 
            # compute the inverse of the full matrix
            Cov=FAInverse(psi,B)
            nu=xt.T.dot(Cov.dot(xt))
        else:
            # use Woodbury formula with old values of B, psi
            p=B.shape[1]
            invM=LA.inv(np.identity(p)+B.T.dot(B/psi))
            U=B.T.dot(xt/psi)
            U2=(xt-B.dot(invM).dot(U))/psi  
            nu=xt.T.dot(U2)
            
        k=RVGALogReg.beta/math.sqrt(nu+RVGALogReg.beta**2)
        a=xt.T.dot(theta)
        
        # Update the covariance
        m=sigmoid(k*a)
        c=k*sigp(k*a)#max(k*sigp(k*a),1e-100)
        return m,c,k
    
    def updateState(self,xt,yt,m,W,psi):
        # Update the state
        error=yt-m
        if psi.all()==0 or psi.any()==0:
            Cov=FAInverse(psi,W)
            theta=self._theta+Cov.dot(xt)*error
        else:
            # use the new updates
            p=W.shape[1]
            invM=LA.inv(np.identity(p)+W.T.dot(W/psi))
            U=W.T.dot(xt/psi)
            theta=self._theta+error*(xt-W.dot(invM).dot(U))/psi 
        return theta
    
    def updateCov(self,xt,c):
        c=max(c,1e-100)
        sigma=np.sqrt(1/c)
        psi,W=self._covAnalyze.fit(xt,sigma,update=False) 
        return psi,W
    
    def update(self,xt,yt):
        
        psi=self.psi
        B=self.B
        theta=self._theta
        if psi.all()==0 or psi.any()==0:
            Cov=self.Cov
            
        # generate samples to compute the expectation
        d=theta.shape[0]
        psi_old=psi
        B_old=B
        
        m,c,k = LargeScaleRVGALogRegExplicit.updateExpectationExact(xt,theta,B,psi)
        
        
        psi,B=self.updateCov(xt,c)
        theta=self.updateState(xt,yt,m,B,psi) 
        
        m,c,k = LargeScaleRVGALogRegExplicit.updateExpectationExact(xt,theta,B,psi)
       
        if self.extragrad and not self.updateCovTwoTimes:
            theta=self.updateState(xt,yt,m,B,psi) 
        if self.extragrad and self.updateCovTwoTimes:
            psi,B=self.updateCov(xt,c)
            theta=self.updateState(xt,yt,m,B,psi) 
                        
        self._covAnalyze._psio=psi
        self._covAnalyze._Wo=B
        self._theta=theta

# The explicit RVGA (without Newton solver) + factor analysis
# The average is computed with sampling using "nbSamples"
# If extragrad=True we update two times the mean 
# If extragrad=True and updateCovTwoTimes=True we update two times the covariance (mirror prox)
class LargeScaleRVGALogRegSampled(LargeScaleBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, psi0, B0, passNumber=1, ppca=False, svd=False,nbInnerLoop=50,nbSamples=10,seed=1,extragrad=True,updateCovTwoTimes=False):
        super().__init__(theta0, psi0, B0, passNumber=passNumber, sigma=1, ppca=ppca, svd=svd,nbInnerLoop=nbInnerLoop)
        np.random.seed(seed)
        d,p=B0.shape
        self.normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamples,))
        self.extragrad=extragrad
        self.updateCovTwoTimes=updateCovTwoTimes
    
    def updateExpectationSampling(self,xt,mu,W,psi):
        thetaVec,pVec=importanceSamples(mu,W,psi,self.normalSamples)
        m=(sigmoid(thetaVec.dot(xt))*pVec).sum()
        c=(sigp(thetaVec.dot(xt))*pVec).sum()
        return m, c
    
    def updateState(self,xt,yt,m,W,psi):
        # Update the state
        error=yt-m
        if psi.all()==0 or psi.any()==0:
            Cov=FAInverse(psi,W)
            theta=self._theta+Cov.dot(xt)*error
        else:
            # use the new updates
            p=W.shape[1]
            invM=LA.inv(np.identity(p)+W.T.dot(W/psi))
            U=W.T.dot(xt/psi)
            theta=self._theta+error*(xt-W.dot(invM).dot(U))/psi 
        return theta
    
    def updateCov(self,xt,c):
        c=max(c,1e-100)
        sigma=np.sqrt(1/c)
        psi,W=self._covAnalyze.fit(xt,sigma,update=False) 
        return psi,W
    
    def update(self,xt,yt):
        
        psi=self.psi
        W=self.B
        theta=self._theta
        if psi.all()==0 or psi.any()==0:
            Cov=self.Cov
            
        # generate samples to compute the expectation
        d=theta.shape[0]
        
        m,c = self.updateExpectationSampling(xt,theta,W,psi)
        
        
        psi,W=self.updateCov(xt,c)
        theta=self.updateState(xt,yt,m,W,psi) 
        
        m,c = self.updateExpectationSampling(xt,theta,W,psi)
        if self.extragrad and not self.updateCovTwoTimes:
            theta=self.updateState(xt,yt,m,W,psi) 
        if self.extragrad and self.updateCovTwoTimes:
            psi,W=self.updateCov(xt,c)
            theta=self.updateState(xt,yt,m,W,psi) 
                
        self._covAnalyze._psio=psi
        self._covAnalyze._Wo=W
        self._theta=theta
        
# Same as previously but with ensembleSamples instead of importanceSamples
class LargeScaleRVGALogRegSampled2(LargeScaleBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, psi0, B0, passNumber=1, ppca=False, svd=False,nbInnerLoop=50,nbSamples=10,seed=1,extragrad=True,updateCovTwoTimes=False):
        super().__init__(theta0, psi0, B0, passNumber=passNumber, sigma=1, ppca=ppca, svd=svd,nbInnerLoop=nbInnerLoop)
        np.random.seed(seed)
        d,p=B0.shape
        self.normalSamplesd=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamples,))
        self.normalSamplesp=np.random.multivariate_normal(np.zeros(p,),np.identity(p),size=(nbSamples,))
        self.extragrad=extragrad
        self.updateCovTwoTimes=updateCovTwoTimes
    
    def updateExpectationSampling(self,xt,mu,W,psi):
        thetaVec=ensembleSamples(mu,W,psi,self.normalSamplesd,self.normalSamplesp).T
        m=(sigmoid(thetaVec.dot(xt))).mean()
        c=(sigp(thetaVec.dot(xt))).mean()
        return m, c
    
    def updateState(self,xt,yt,m,W,psi):
        # Update the state
        error=yt-m
        if psi.all()==0 or psi.any()==0:
            Cov=FAInverse(psi,W)
            theta=self._theta+Cov.dot(xt)*error
        else:
            # use the new updates
            p=W.shape[1]
            invM=LA.inv(np.identity(p)+W.T.dot(W/psi))
            U=W.T.dot(xt/psi)
            theta=self._theta+error*(xt-W.dot(invM).dot(U))/psi 
        return theta
    
    def updateCov(self,xt,c):
        c=max(c,1e-100)
        sigma=np.sqrt(1/c)
        psi,W=self._covAnalyze.fit(xt,sigma,update=False) 
        return psi,W
    
    def update(self,xt,yt):
        
        psi=self.psi
        W=self.B
        theta=self._theta
        if psi.all()==0 or psi.any()==0:
            Cov=self.Cov
            
        # generate samples to compute the expectation
        d=theta.shape[0]
        
        m,c = self.updateExpectationSampling(xt,theta,W,psi)
        
        
        psi,W=self.updateCov(xt,c)
        theta=self.updateState(xt,yt,m,W,psi) 
        
        m,c = self.updateExpectationSampling(xt,theta,W,psi)
        if self.extragrad and not self.updateCovTwoTimes:
            theta=self.updateState(xt,yt,m,W,psi) 
        if self.extragrad and self.updateCovTwoTimes:
            psi,W=self.updateCov(xt,c)
            theta=self.updateState(xt,yt,m,W,psi) 
                
        self._covAnalyze._psio=psi
        self._covAnalyze._Wo=W
        self._theta=theta
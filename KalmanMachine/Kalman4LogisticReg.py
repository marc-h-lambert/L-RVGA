###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Online second order method for logistic regression :                            #
# The extended Kalman filter = online natural gradient                            #
# --> see "Online natural gradient as a Kalman filter, Yann Olivier 2018"         #                
# The quadratic Kalman filter= online version of the bounded variational approach #
# --> see "A variational approach to Bayesian logistic regression models \        #
#     and their extensions, Jaakkola and Jordan 1997"                             #
# The recursive VGA implicit                                                      #
# The recursive VGA explicit                                                      #
# --> see "The recursive variational Gaussian approximation (R-VGA),              #
#     Marc Lambert, Silvere Bonnabel and Francis Bach 2020"                       #
# The recursive VGA explicit with extragrad                                       #
# --> see "The limited memory recursive variational Gaussian approximation(L-RVGA)#              
#     Marc Lambert, Silvere Bonnabel and Francis Bach 2021"                       #                                                                                                                  
###################################################################################

from .KUtils import sigmoid, sigp, sigpp, graphix, negbayesianlogisticPdf
import numpy as np
import numpy.random
import numpy.linalg as LA
from .KBayesianReg import BayesianRegression, OnlineBayesianRegression
import math
from math import log, exp
from scipy import optimize

class LogisticPredictor(object):
    
    def __init__(self):
        super().__init__()
        
    def predict(self,X):
        return np.multiply(self.predict_proba(X)>0.5,1)
    
    #prediction of N outputs for inputs X=(N,d)
    #use approximation of the integration over a Gaussian
    def predict_proba(self,X):
        N,d=X.shape
        beta=math.sqrt(8/math.pi)
        vec_nu=np.diag(X.dot(self.Cov).dot(X.T))
        k=beta/np.sqrt(vec_nu+beta**2).reshape(N,1)
        return sigmoid(k*X.dot(self._theta))
    
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

import time, tracemalloc

# Batch Laplace version of Bayesian Logistic Regression (with a Gaussian model)
class LaplaceLogisticRegression(BayesianRegression, LogisticPredictor):
    
    # !!! implement only for theta0=0 and Cov0=sigma0^2 I, 
    # otherwise using sikit.logreg method produce biased maximum posterior
    def __init__(self, theta0, Cov0):
        super().__init__(theta0)
        self._Cov0=Cov0 # the initial covariance (ie uncertainty on the initial guess)
        self._Cov=np.copy(self._Cov0)

    def fit(self,X,y):   
        N,d=X.shape
        (sign, logdetCov) = LA.slogdet(self._Cov0)
        logdetCov0=logdetCov
        invCov0=LA.inv(self._Cov0)
        
        tracemalloc.start()
        tic=time.perf_counter()  
        sol=optimize.minimize(negbayesianlogisticPdf, self._theta0, args=(self._theta0,invCov0,logdetCov0,X,y.reshape(N,),1,),method='L-BFGS-B')
        self._theta=sol.x
        
        toc=time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memoryUsedLAP=current/10**6
        timeCostLAP=toc-tic
        print('Compute MAP with sikkit learn/LBFGS... in {0:.2} s'.format(timeCostLAP))
        print('Memory cost for MAP with sikkit learn/LBFGS... is {0:.2} MB'.format(memoryUsedLAP))
        
        tracemalloc.start()
        tic=time.perf_counter()  
        
        # the Hessian 
        L=sigmoid(X.dot(self._theta))
        K=(L*(1-L)).reshape(N,1,1)
        # Tensor version
        #A=X[...,None]*X[:,None]
        #H=np.sum(K*A,axis=0)+LA.inv(self._Cov0)
        # Memory free version
        H=invCov0
        for i in range(0,N):
            xt=X[i,:].reshape(d,1)
            H=H+K[i]*xt.dot(xt.T)
        self._Cov=LA.inv(H)
        
        toc=time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.memoryUsedLAP=current/10**6
        self.timeCostLAP=toc-tic
        print('Compute LAP with inversion of Hessian... in {0:.2} s'.format(self.timeCostLAP))
        print('Memory cost for LAP ... is {0:.2} MB'.format(self.memoryUsedLAP))
        
        return self
    
    def plotEllipsoid(self,ax,nbLevels=1,u=0,v=1,labelize=True):
        d=self._theta.shape[0]
        thetaproj,Covproj=graphix.projEllipsoid(self._theta,self._Cov.reshape(d,d),u,v)
        if labelize:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,col='r',linewidth=1.2,zorder=3,linestyle='-',label='Laplace')
        else:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,col='r',linewidth=1.2,zorder=3,linestyle='-')
        ax.scatter(self._theta[0],self._theta[1],color='r')
        
    @property
    def Cov(self):
        return self._Cov
    
    @property
    def maxP(self):
        return self._theta

# The natural gradient or EKF
class EKFLogReg(OnlineBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, Cov0, sigma=1, passNumber=1):
        super().__init__(theta0, Cov0, sigma, passNumber)
    
    def update(self,xt,yt):
        # intermediate variables
        nu=xt.T.dot(self._Cov.dot(xt))
        Pu=self._Cov.dot(xt)
            
        a=xt.T.dot(self._theta)
        
        m=sigp(a)
        m=max(m,1e-100)
        
        # update state
        self._Cov=self._Cov-np.outer(Pu,Pu)/(1/m+nu)
        self._theta=self._theta+self._Cov.dot(xt)*(yt-sigmoid(a))
        
    # equivalent formulas using the Kalman form
    # def update(self,xt,yt):
    #     # compute R
    #     mu=sigmoid(xt.T.dot(self._theta)) 
    #     R=max(mu*(1-mu),1e-100)
    #     H=R*xt.T

    #     # prediction error
    #     err=yt-mu
        
    #     # computation of optimal gain
    #     S=R+H.dot(self._Cov).dot(H.T)
    #     K=self._Cov.dot(H.T).dot(LA.inv(S))
        
    #     # update state and covariance of state
    #     self._theta=self._theta+K.dot(err)
    #     self._Cov=self._Cov-K.dot(H).dot(self._Cov)
    

# The local variational Kalman or quadratic Kalman      
class QKFLogReg(OnlineBayesianRegression,LogisticPredictor):
    
    def __init__(self, theta0, Cov0, sigma=1, passNumber=1):
        super().__init__(theta0, Cov0, sigma, passNumber)
    
    @staticmethod
    def eta(x):
        return -1/(2*x)*(sigmoid(x)-0.5)

    def update(self,xt,yt):
        # compute matrix R
        ksi=math.sqrt(xt.T.dot(self._Cov+np.outer(self._theta,self._theta)).dot(xt))
        invR=np.ones([1,1])*(-2*QKFLogReg.eta(ksi))
        R=LA.inv(invR)
            
        # compute gain K
        H=xt.T
        S=R+H.dot(self._Cov).dot(H.T)
        K=self._Cov.dot(H.T).dot(LA.inv(S))
                            
        #update theta
        self._theta=self._theta+K.dot(R.dot(yt-0.5)-H.dot(self._theta))
                
        #update Cov
        self._Cov=self._Cov-K.dot(H).dot(self._Cov)
      
    # equivalent formulas from Jordan paper:
    # def update(self,xt,yt):
    #     ksi=math.sqrt(xt.T.dot(self._Cov+np.outer(self._theta,self._theta)).dot(xt))
    #     print(ksi)
    #     P=LA.inv(LA.inv(self._Cov)-2*QKFLogReg2.eta(ksi)*np.outer(xt,xt))
        
    #     self._theta=P.dot(LA.inv(self._Cov).dot(self._theta)+(yt-0.5)*xt)
    #     self._Cov=P
            

# The implicit RVGA (with a Newton solver)
class RVGALogReg(OnlineBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, Cov0, sigma=1,passNumber=1):
        super().__init__(theta0, Cov0, sigma, passNumber)
    
    beta=math.sqrt(8/math.pi)
            
    def fun2D(x,alpha0,nu0,y):
        k=RVGALogReg.beta/math.sqrt(exp(x[1])+RVGALogReg.beta**2)
        f=x[0]+nu0*sigmoid(x[0]*k)-alpha0-nu0*y
        g=exp(x[1])-nu0/(1+nu0*k*sigp(x[0]*k))
        return [f,g]

    def jac(x,alpha0,nu0,y):
        k=RVGALogReg.beta/math.sqrt(exp(x[1])+RVGALogReg.beta**2)
        kp=-0.5*RVGALogReg.beta*exp(x[1])/((exp(x[1])+RVGALogReg.beta**2)**(3/2))
        f_a=1+nu0*k*sigp(x[0]*k)
        f_gamma=nu0*x[0]*kp*sigp(x[0]*k)
        g_a=nu0**2*k**2*sigpp(x[0]*k)/((1+nu0*k*sigp(x[0]*k))**2)
        g_gamma=exp(x[1])+nu0**2*kp*(sigp(x[0]*k)+k*x[0]*sigpp(x[0]*k))/((1+nu0*k*sigp(x[0]*k))**2)
        return np.array([[f_a,f_gamma],[g_a,g_gamma]])
        
    def optim2D(alpha0,nu0,y):
        
        alphaMin=alpha0+nu0*y-nu0
        alphaMax=alpha0+nu0*y
        nuMin=nu0*(1-nu0/(4+nu0))
        nuMax=nu0
        a=(alphaMin+alphaMax)/2
        gamma=log((nuMin+nuMax)/2)
        #a,nu=alpha0,nu0
        sol=optimize.root(RVGALogReg.fun2D, [a,gamma], tol=1e-6, args=(alpha0,nu0,y,),jac=RVGALogReg.jac,method='hybr')

        return sol.x[0],exp(sol.x[1]) ,sol.nfev
    
    def update(self,xt,yt):
        
        # init parameters 
        nu0=xt.T.dot(self._Cov.dot(xt))
        alpha0=xt.T.dot(self._theta)
            
        a,nu,nbInnerLoop=RVGALogReg.optim2D(np.asscalar(alpha0), np.asscalar(nu0), np.asscalar(yt))
        
        #updates
        k=RVGALogReg.beta/math.sqrt(nu+RVGALogReg.beta**2)
        self._theta=self._theta+self._Cov.dot(xt)*(yt-sigmoid(k*a))
        s=1/(nu0+1/(k*sigp(k*a)))
        self._Cov=self._Cov-s*np.outer(self._Cov.dot(xt),self._Cov.dot(xt))

# The explicit RVGA (without Newton solver)
class RVGALogRegExplicit(OnlineBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, Cov0, sigma=1, passNumber=1):
        super().__init__(theta0, Cov0, sigma, passNumber)
    
    def update(self,xt,yt):
        beta=RVGALogReg.beta # beta = math.sqrt(8/math.pi)
        
        # intermediate variables
        nu=xt.T.dot(self._Cov.dot(xt))
            
        # compute sigma(a)
        k=beta/math.sqrt(nu+beta**2)
        a=xt.T.dot(self._theta)
        
        m=k*sigp(k*a)
        m=max(m,1e-100)
        
        # update covariance
        Pu=self._Cov.dot(xt)
        self._Cov=self._Cov-np.outer(Pu,Pu)/(1/m+nu)
        self._theta=self._theta+self._Cov.dot(xt)*(yt-sigmoid(k*a))
        # update state

# The explicit RVGA with one extragrad (akka Mirror prox)
# the mean is always updated two times, the covariance is updated two times 
# if updateCovTwoTimes=True
class RVGALogRegIterated(OnlineBayesianRegression, LogisticPredictor):
    
    def __init__(self, theta0, Cov0, sigma=1, passNumber=1,updateCovTwoTimes=True):
        super().__init__(theta0, Cov0, sigma, passNumber)
        self.updateCovTwoTimes=updateCovTwoTimes
    
    @staticmethod
    def updateExpectationParameters(xt,theta,Cov):
        beta=RVGALogReg.beta
        nu=xt.T.dot(Cov.dot(xt))
        k=beta/math.sqrt(nu+beta**2)
        
        a=xt.T.dot(theta)
        return k,a,nu
                
    def update(self,xt,yt):
        Cov=self._Cov
        theta=self._theta
        nuOld=xt.T.dot(self._Cov.dot(xt))
        PuOld=self._Cov.dot(xt)
        k,a,nu=RVGALogRegIterated.updateExpectationParameters(xt,theta,Cov)
        
        # update covariance: may be transfer in the loop but pose problem on the 
        # condition number
        m=k*sigp(k*a)
        if m<1e-100:
            print(m)
        m=max(m,1e-100)
        
        # update state
        Cov=self._Cov-np.outer(PuOld,PuOld)/(1/m+nuOld)
        theta=self._theta+Cov.dot(xt)*(yt-sigmoid(k*a))
        k,a,nu=RVGALogRegIterated.updateExpectationParameters(xt,theta,Cov)
        m=k*sigp(k*a)
        if m<1e-100:
            m=max(m,1e-100)
        if self.updateCovTwoTimes:
            Cov=self._Cov-np.outer(PuOld,PuOld)/(1/m+nuOld)
        theta=self._theta+Cov.dot(xt)*(yt-sigmoid(k*a))

        self._Cov=Cov
        self._theta=theta
        
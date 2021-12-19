###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Factor analysis                                                                 #
###################################################################################

import numpy as np
import numpy.random
import numpy.linalg as LA
import math

class CovarianceFactorAnalysis(object):

    def __init__(self, psio, Wo, ppca=False):
        self._psio = psio
        self._Wo = Wo
        self._ppca=ppca
                
    def fit(self,xi,sigma=1):
        pass
    
    @property
    def psi(self):
        return self._psio
    
    @property
    def W(self):
        return self._Wo
    
    @property
    def faCov(self):
        d=self._Wo.shape[0]
        cov=np.diag(self.psi.reshape(d,))+self.W.dot(self.W.T)
        return cov
    
    
    
# Covariance factor Analysis with recursive EM : version used for RVGA variational regression
class CovarianceFactorAnalysisEM(CovarianceFactorAnalysis):

    def __init__(self, psio, Wo, ppca=False,nbInnerLoop=10,fixedPoint=False):
        super().__init__(psio, Wo,ppca)
        self._fixedPoint=fixedPoint
        self._nbInnerLoop = nbInnerLoop

    def fit(self,xi,sigma=1,update=True):
        n_features, size_batchs = xi.shape
        n_components = self._Wo.shape[1]
        xi=xi.T/sigma
        
        psio=self._psio
        Wo=self._Wo
        psi=self._psio
        W=self._Wo
                
        diagWoWo=np.sum(Wo*Wo,axis=1).reshape([n_features,1])
        diagXiXi=np.sum(xi*xi,axis=0).reshape([n_features,1])
        diagS=diagXiXi+diagWoWo+psio

        for j in range(0,self._nbInnerLoop):
            invPsiW=W/psi 
            invM=LA.inv(np.identity(n_components)+W.T.dot(invPsiW))
            
            # compute xi terms
            x_invPsi_W=(xi/(psi.T)).dot(W) 
            x_invPsi_W=xi.T.dot(x_invPsi_W)

            # compute Wo terms
            Wo_invPsi_W=(Wo.T/psi.T).dot(W) 
            Wo_invPsi_W=Wo.dot(Wo_invPsi_W)

            # compute psio terms
            psio_invPsi_W=W*(psio/psi)

            S_invPsi_W=x_invPsi_W+Wo_invPsi_W+psio_invPsi_W

            # compute intermediate terms
            A=np.identity(n_components)+invM.dot(W.T).dot(S_invPsi_W/psi)
            Wnew=S_invPsi_W.dot(LA.inv(A))
            W1=Wnew.dot(invM)
            W2=S_invPsi_W
            diagW1W2=np.sum(W1*W2,axis=1).reshape([n_features,1])
            psiNew=diagS-diagW1W2
            
            W=Wnew
            psi=psiNew
            
            if self._ppca:
                psi=np.ones([n_features,1]) *psi.sum()/n_features
        
        if update:
            self._psio=psi
            self._Wo=W

        return psi,W 
    
# Covariance factor Analysis with recursive EM : version used for covariance approximation
# the difference with CovarianceFactorAnalysisEM is that an online average is computed
class CovarianceFactorAnalysisRecEM(CovarianceFactorAnalysis):

    def __init__(self, psio, Wo, ppca=False,nbInnerLoop=10,fixedPoint=False):
        super().__init__(psio, Wo,ppca)
        self._fixedPoint=fixedPoint
        self._nbInnerLoop = nbInnerLoop

    def fit(self,xi,t,update=True,firstInput=True):
        size_batchs, n_features = xi.shape
        n_components = self._Wo.shape[1]
        
        if firstInput:
            psio=np.zeros([n_features,1])
            Wo=np.zeros([n_features,n_components])
        else:
            psio=self._psio
            Wo=self._Wo
        psi=self._psio
        W=self._Wo
        
        #S=xi.T.dot(xi)+Wo.dot(Wo.T)+np.diag(psio.reshape(-1,))
        gamma=1/(t+1)
        
        diagWoWo=np.sum(Wo*Wo,axis=1).reshape([n_features,1])
        diagXiXi=np.sum(xi*xi,axis=0).reshape([n_features,1])
        diagS=gamma*diagXiXi+(1-gamma)*(diagWoWo+psio)

        for j in range(0,self._nbInnerLoop):
            invPsiW=W/psi 
            invM=LA.inv(np.identity(n_components)+W.T.dot(invPsiW))
            
            # compute xi terms
            x_invPsi_W=(xi/(psi.T)).dot(W) 
            x_invPsi_W=xi.T.dot(x_invPsi_W)

            # compute Wo terms
            Wo_invPsi_W=(Wo.T/psi.T).dot(W) 
            Wo_invPsi_W=Wo.dot(Wo_invPsi_W)

            # compute psio terms
            psio_invPsi_W=W*(psio/psi)

            S_invPsi_W=gamma*x_invPsi_W+(1-gamma)*(Wo_invPsi_W+psio_invPsi_W)
            
            # compute intermediate terms
            A=np.identity(n_components)+invM.dot(W.T).dot(S_invPsi_W/psi)
            Wnew=S_invPsi_W.dot(LA.inv(A))
            W1=Wnew.dot(invM)
            W2=S_invPsi_W
            diagW1W2=np.sum(W1*W2,axis=1).reshape([n_features,1])
            psiNew=diagS-diagW1W2
            
            W=Wnew
            psi=psiNew
            
            if self._ppca:
                psi=np.ones([n_features,1]) *psi.sum()/n_features
        
        if update:
            self._psio=psi
            self._Wo=W
        return psi,W 

# Covariance factor Analysis with online EM (Cappé & Moulines)
class CovarianceFactorAnalysisOnlineEM(CovarianceFactorAnalysis):

    def __init__(self, psio, Wo, ppca=False,nbInnerLoop=10,fixedPoint=False,averaging=False):
        super().__init__(psio, Wo,ppca)
        d,p = Wo.shape
        self._fixedPoint=fixedPoint
        self._nbInnerLoop = nbInnerLoop
        invPsiW0=Wo/psio 
        self._invM0=LA.inv(np.identity(p)+Wo.T.dot(invPsiW0)) 
            
        # suffiscient statistics
        self._mu=np.zeros([p,d])
        self._P=np.zeros([p,p])
        self._diagS=np.zeros([d,1])
        self._averaging=averaging
        
    def gamma (self,t):
        if t==0:
            return 1
        else:
            return t**(-0.6)#1/math.sqrt(t)
        
    def fit(self,xi,t,sigma=1,update=True):
        n_features, size_batchs = xi.shape
        n_components = self._Wo.shape[1]
        
        psi=self._psio
        W=self._Wo
        
        for j in range(0,1):
            invPsiW=W/psi 
            invM=LA.inv(np.identity(n_components)+W.T.dot(invPsiW)) 
            invPsix=xi/psi
            Ez=invM.dot(W.T).dot(invPsix)

            # update suffiscient statistics
            self._mu=(1-self.gamma(t))*self._mu+self.gamma(t)*Ez.dot(xi.T)
            self._P=(1-self.gamma(t))*self._P+self.gamma(t)*(invM+Ez.dot(Ez.T))
            diagXiXi=np.sum(xi*xi,axis=1).reshape([n_features,1])
            self._diagS=(1-self.gamma(t))*self._diagS+self.gamma(t)*diagXiXi
                        
            # compute intermediate terms
            W=self._mu.T.dot(LA.inv(self._P))#+self._invM0/(t+1)))
            #W=self._mu.T.dot(LA.inv(self._P))
            psi=self._diagS-np.sum(W*self._mu.T,axis=1).reshape([n_features,1])
                        
            if self._ppca:
                psi=np.ones([n_features,1]) *psi.sum()/n_features
        
        if update:
            if self._averaging and t>500:#Polyak averaging
                t0=t-500
                self._psio=self._psio*(t0-1)/t0+psi/t0
                self._Wo=self._Wo*(t0-1)/t0+W/t0
            else:
                self._psio=psi
                self._Wo=W
        return psi,W 
        
# Covariance factor Analysis with batch EM : full matrix version
class CovarianceFactorAnalysisEM_batch(CovarianceFactorAnalysis):

    def __init__(self, psio, Wo, ppca=False,nbInnerLoop=10,fixedPoint=False):
        super().__init__(psio, Wo,ppca)
        self._fixedPoint=fixedPoint
        self._nbInnerLoop = nbInnerLoop

                
    def fit(self,S,p):
        d = S.shape[0]

        psi=self._psio
        W=self._Wo
        
        #for j in range(0,self._nbInnerLoop): #the caller manage iterations 
        invPsiW=W/(psi.reshape(-1,1))
        invM=LA.inv(np.identity(p)+W.T.dot(invPsiW))
        A=LA.inv(np.identity(p)+invM.dot(invPsiW.T).dot(S).dot(invPsiW))
        Wnew=S.dot(invPsiW).dot(A)
        psiNew=np.diag(S-Wnew.dot(invM).dot(invPsiW.T).dot(S)).reshape(d,1)
        W=Wnew
        psi=psiNew
        if self._ppca:
            psi=np.ones([d,1]) *psi.sum()/d
                
        self._psio=psi
        self._Wo=W
        return psi,W     

# Covariance factor Analysis with SVD
import sys  
def sortedEigs(A,p,ascent=True):
    l,v=LA.eigh(A)
    max_index=sorted(range(len(l)),key=lambda k: l[k],reverse=not ascent)
    if p <= len(l):
        max_index=max_index[0:p]
    else:
        print('p > nb eigens')
        sys.exit(1) 
    return l[max_index],v[:,max_index]

def PPCAapprox(S,p):
    d=S.shape[0]
    l,U=sortedEigs(S,d,ascent=False) 
    if d==p:
        sigma2=0
    else:
        sigma2=np.abs(np.sum(l[p:])/(d-p))
    lmax=l[0:p]
    mui=[0 if lmax[i]< sigma2 else math.sqrt(lmax[i]-sigma2) for i in range(0,len(lmax))]
    return math.sqrt(sigma2), U[:,0:p].dot(np.diag(mui))

class CovarianceFactorAnalysisSVD(CovarianceFactorAnalysis):

    def __init__(self, psio, Wo, ppca=True):
        super().__init__(psio, Wo, ppca)
                
    def fit(self,xi,sigma=1):
        n_features, size_batchs = xi.shape
        n_components = self._Wo.shape[1]
        invP=self._Wo.dot(self._Wo.T)+np.diag(self._psio.reshape(n_features,))+xi.dot(xi.T)/sigma**2
        alpha,B=PPCAapprox(invP,n_components)
        self._psio=alpha**2*np.ones([n_features,1])
        self._Wo=B
        return self._psio, self._Wo

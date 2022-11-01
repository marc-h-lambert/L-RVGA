###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Mathematical functions                                                          #
###################################################################################

import numpy.linalg as LA 
import numpy as np
import math
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt

############## Graphix tools ########################## 
# the logistic function
def sigmoid(x):
    x=np.clip(x,-100,100)
    return 1/(1+np.exp(-x))

# the first derivative of the logistic function
def sigp(x):
    return sigmoid(x)*(1-sigmoid(x))

# the second derivative of the logistic function
def sigpp(x):
    return sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))

# a quadratic form
def quadratic2D(x, y, H,v,c):
    q11=H[0,0]
    q22=H[1,1]
    q12=H[0,1]
    v1=v[0]
    v2=v[1]
    return 0.5*q11*x**2 + 0.5*q22*y**2 + q12*x*y-v1*x-v2*y+c

# the likelihood density for logistic regression, Y supposed in {0,1} 
def logisticPdf(theta,X,Y,Z):
    N,d=X.shape
    theta=theta.reshape(d,1)
    # we use the relation logloss=sigmoid(y.theta.u)
    # and log sigmoid(y.theta.u) = - log(1+exp(-y.theta.u)=- logexp(0,-y.theta.u)
    Yb=2*Y.reshape(N,1)-1
    log_pdf_likelihood=np.sum(-np.logaddexp(0, -Yb*X.dot(theta)),axis=0)
    return log_pdf_likelihood-math.log(Z)

# the bayesian density for logistic regression, Y supposed in {0,1}   
def bayesianlogisticPdf(theta,theta0, invCov0,lodetCov0,X,Y,Z):
    N,d=X.shape
    # compute log prior:
    theta=theta.reshape(d,1)
    theta0=theta0.reshape(d,1)
    e=(theta-theta0).reshape(d,1)
    log_pdf_prior=-0.5*(e.T.dot(invCov0).dot(e))-0.5*d*math.log(2*math.pi)+0.5*lodetCov0
    return logisticPdf(theta,X,Y,Z)+log_pdf_prior

# the bayesian density for logistic regression, Y supposed in {0,1}   
def bayesianlogisticPdf_largeScale(theta,theta0, W0,psi0,lodetP0,X,Y,Z):
    N,d=X.shape
    # compute log prior:
    e=(theta-theta0).reshape(d,1)
    We=W0.T.dot(e)
    ePsie=e.T.dot(e*psi0)
    log_pdf_prior=-0.5*(ePsie+We.T.dot(We))-0.5*d*math.log(2*math.pi)+0.5*lodetP0
    return logisticPdf(theta,X,Y,Z)+log_pdf_prior

def KLDivergence(P0,P1):
    d=P0.shape[0]
    (sign, logdetP0) = LA.slogdet(P0)
    if sign <0:
        print("logdet <0 for P0 =",P0)
    (sign, logdetP1) = LA.slogdet(P1)
    if sign <0:
        print("logdet <0 for P1=",P1)
    return 0.5*(logdetP1-logdetP0+np.trace(LA.inv(P1).dot(P0))-d)

def KLDivergence_largeScale(P0,logdetP0,psi,W):
    d=P0.shape[0]
    invP1=FAInverse(psi,W)  
    logdetP1=fastLogDet(psi,W)
    return 0.5*(logdetP1-logdetP0+np.trace(invP1.dot(P0))-d)

def negbayesianlogisticPdf(theta,theta0, invCov0,lodetCov0,X,Y,Z):
    return -bayesianlogisticPdf(theta,theta0, invCov0,lodetCov0,X,Y,Z).reshape(1,)

def neglogisticPdf(theta,X,Y,Z):
    return -logisticPdf(theta,X,Y,Z)

#if U N x d, p N x 1 compute Sum pi Ui Ui^T of size d x d
def empiricalCov(U,p):
    M=U[...,None]*U[:,None]
    p=p.reshape(-1,1,1)
    return np.sum(p*M,axis=0)


def FAInverse(psi,B):
    if psi.all() != 0:
        d,p=B.shape
        psi=psi.reshape(d,1)
        invM=LA.inv(np.identity(p)+B.T.dot(B/psi))
        return (np.identity(d)-B.dot(invM).dot(B.T/psi.T))/psi
    else:
        return LA.inv(B.dot(B.T))
    
# sample Gaussian variable with respect to N(mu,inv(WW+Diag(psi))) in a fast way
# we suppose we have already N samples following N(O,Id) in the variable 
# "normalSamplesd" of shape N x d
# return the new samples xi and the weights pi
def importanceSamples(mu,W,psi,normalSamplesd):
    d,p=W.shape
    psi=psi.reshape(d,1)
    xi=normalSamplesd.T/np.sqrt(psi) # x sim N(O,invPsi)
    xi=xi.T
    k=xi.dot(W) # z|x sim N(W^Tx,I) --> we consider z=0
    pi=(2*math.pi)**(-p/2)*np.exp(-0.5*np.diag(k.dot(k.T)))
    pi=pi.reshape(-1,1)
    pi=pi/pi.sum(0)
    xi=mu.T+xi # add the mean
    return xi, pi

# sample Gaussian variable with respect to N(mu,inv(WW+Diag(psi))) in a fast way
# we suppose we have already N samples following N(O,Id) in the variable 
# "normalSamplesd" of shape N x d
# return the new samples xi and the weights pi
def ensembleSamples(mu,W,psi,normalSamplesd, normalSamplesp):
    d,p=W.shape
    psi=psi.reshape(d,1)
    xi=normalSamplesd.T/np.sqrt(psi) # x sim N(O,invPsi)
    eps=normalSamplesp
    M=np.identity(p)+W.T.dot(W/psi)
    L=(W/psi).dot(LA.inv(M))
    x=xi-L.dot(W.T.dot(xi))+L.dot(eps.T)
    return mu+x

# compute logDet(Diag(psi)+WW^T) using the matrix determinant Lemma
def fastLogDet(psi,W):
    d,p=W.shape
    if d==p: # full rank: nothing to do
        M=np.diag(psi.reshape(d,))+W.dot(W.T)
        (sign, logdetM) = LA.slogdet(M)
        return logdetM
    else:
        if psi.any() != 0:
            lodetPsi=np.log(psi).sum()
            M=np.eye(p)+(W.T/psi.T).dot(W)
            (sign, logdetM) = LA.slogdet(M)
            return lodetPsi+logdetM
        else: 
            print("error in fastLogDet: covariance not definite")
    
class graphix: 
    # plot a 2D ellipsoid
    def plot_ellipsoid2d(ax,origin,Cov,col='r',zorder=1,label='',linestyle='dashed',linewidth=1):
        L=LA.cholesky(Cov)
        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        x = np.cos(theta)
        y = np.sin(theta)
        x,y=origin.reshape(2,1) + L.dot([x, y])
        ax.plot(x, y,linestyle=linestyle,color=col,zorder=zorder,label=label,linewidth=linewidth)
    
    # project a ND ellipsoid (mean-covariance) in plane (i,j)
    def projEllipsoid(theta,P,i,j):
        thetaproj=np.array([theta[i],theta[j]])
        Pproj=np.zeros((2,2))
        Pproj[0,0]=P[i,i]
        Pproj[0,1]=P[i,j]
        Pproj[1,0]=P[j,i]
        Pproj[1,1]=P[j,j]
        return thetaproj,Pproj
    
    def projEllipsoidOnVector(theta,P,v1,v2):
        x1=v1.T.dot(theta)
        x2=v2.T.dot(theta)
        thetaproj=np.array([x1,x2])
        v11=v1.T.dot(P).dot(v1)
        v22=v2.T.dot(P).dot(v2)
        v12=v1.T.dot(P).dot(v2)
        Pproj=np.array([[v11,v12], [v12,v22]])
        return thetaproj,Pproj
    
    
if __name__=="__main__":
    Test=['Sampling3']
    if 'Sampling' in Test:
        # test fast sampling
        d=10000
        p=10
        M=10
        normalSamplesd=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(M,))
        normalSamplesp=np.random.multivariate_normal(np.zeros(p,),np.identity(p),size=(M,))
        
        np.random.seed(1)
        psi=np.arange(1,d+1).reshape(d,1)
        W=np.random.uniform(size=[d,p])
        mu=np.random.uniform(size=[d,1])
        TrueCov=FAInverse(psi,W)
        print("True inv(Psi+WW)=\n {0}".format(np.trace(TrueCov)))
        
        # Sampling with full matrix
        thetaVec=np.linalg.cholesky(TrueCov).dot(normalSamplesd.T)
        Cov=empiricalCov(thetaVec.T,np.ones([M,1])/M)
        print("Empirical Cov (full sampling)= \n {0}".format(np.trace(Cov)))
        
        # Importance Sampling
        xi,pi=importanceSamples(np.zeros([d,1]),W,psi,normalSamplesd)
        Cov=empiricalCov(xi,pi)
        print("Empirical Cov (importance sampling v1)= \n {0}".format(np.trace(Cov)))
        
        # Ensemble Sampling
        xi=ensembleSamples(np.zeros([d,1]),W,psi,normalSamplesd,normalSamplesp)
        C=xi[...,None]*xi[:,None]
        Cov= np.sum(C,axis=0)/M
        print("Empirical Cov (ensemble sampling v1)= \n {0}".format(np.trace(Cov)))
        
        
    if 'Sampling2' in Test:
        # test fast sampling
        d=1000
        p=10
        M=1000
        normalSamplesd=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(M,))
        normalSamplesp=np.random.multivariate_normal(np.zeros(p,),np.identity(p),size=(M,))
        
        np.random.seed(1)
        psi=np.arange(1,d+1).reshape(d,1)
        W=np.random.uniform(size=[d,p])
        mu=np.random.uniform(size=[d,1])
        TrueCov=FAInverse(psi,W)
        
        # Sampling with full matrix
        S=0
        thetaVec=mu+np.linalg.cholesky(TrueCov).dot(normalSamplesd.T)
        for i in range(0,M):
            thetai=thetaVec[:,i].reshape(d,1)
            S=S-np.log(thetai.T.dot(thetai))
        S=S/M
        print("estimation of S (full sampling)={0}".format(S))
        
        # Importance Sampling 
        S=0
        thetaVec,pVec=importanceSamples(mu,W,psi,normalSamplesd)
        for i in range(0,M):
            thetai=thetaVec[i,:].reshape(d,1)
            S=S-pVec[i]*np.log(thetai.T.dot(thetai))
        print("estimation of S (Importance sampling)={0}".format(S))
        
        # Ensemble Sampling 
        S=0
        thetaVec=ensembleSamples(mu,W,psi,normalSamplesd,normalSamplesp)
        for i in range(0,M):
            thetai=thetaVec[i,:].reshape(d,1)
            S=S-np.log(thetai.T.dot(thetai))/M
        print("estimation of S (Ensemble sampling)={0}".format(S))
        
    if 'Sampling3' in Test:
        # test fast sampling
        d=10000
        p=10
        M=10
        np.random.seed(1)
        normalSamplesd=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(M,))
        xt=np.random.uniform(0,1,size=d).reshape(d,1).T#np.random.multivariate_normal(np.random.uniform(d,),np.identity(d),size=(1,))
        xt=5*xt/d
        normalSamplesp=np.random.multivariate_normal(np.zeros(p,),np.identity(p),size=(M,))
        
        np.random.seed(1)
        psi=np.arange(1,d+1).reshape(d,1)
        W=np.random.uniform(size=[d,p])
        mu=np.random.uniform(size=[d,1])
        TrueCov=FAInverse(psi,W)
        
        # Analytic expression with Inverse probit 
        beta=math.sqrt(8/math.pi)
        nu=xt.dot(TrueCov.dot(xt.T))
        k=beta/math.sqrt(nu+beta**2)
        m=sigmoid(k*xt.dot(mu))[0][0]
        c=sigp(k*xt.dot(mu))[0][0]
        print("logistic Expectation (exact) \n m={0} c={1} \n".format(m,c))
        
        # Sampling with full matrix
        thetaVec=mu+np.linalg.cholesky(TrueCov).dot(normalSamplesd.T)
        m=(sigmoid(xt.dot(thetaVec))).sum()/M
        c=(sigp(xt.dot(thetaVec))).sum()/M
        print("logistic Expectation  (Full sampl.) \n m={0} c={1} \n".format(m,c))
        
        # Importance Sampling 
        S=0
        thetaVec,pVec=importanceSamples(mu,W,psi,normalSamplesd)
        m=(sigmoid(xt.dot(thetaVec.T))*pVec).sum()/M
        c=(sigp(xt.dot(thetaVec.T))*pVec).sum()/M
        print("logistic Expectation (Importance sampl.) \n m={0} c={1} \n".format(m,c))
        
        # Ensemble Sampling 
        S=0
        thetaVec=ensembleSamples(mu,W,psi,normalSamplesd,normalSamplesp)
        m=(sigmoid(xt.dot(thetaVec))).sum()/M
        c=(sigp(xt.dot(thetaVec))).sum()/M
        print("logistic Expectation  (Ensemble sampl.) \n m={0} c={1} \n".format(m,c))
                
    if 'FastLogeDet' in Test:
        d=5
        p=1
        psi=np.random.uniform(size=[d,1])*0.1
        W=np.random.uniform(size=[d,p])
        
        Cov=np.diag(psi.reshape(d,))+W.dot(W.T)
        (sign, logdet) = LA.slogdet(Cov)
        print("Log Det exact={0}".format(logdet))
        
        logdetApprox=fastLogDet(psi,W)
        print("Log Det approximated={0}".format(logdetApprox))
        
    if 'Posterior' in Test:
        # test fast sampling
        d=5
        p=1
        psi0=np.arange(1,d+1).reshape(d,1)
        W0=np.random.uniform(size=[d,p])
        mu0=np.random.uniform(size=[d,1])
        Cov0=FAInverse(psi0,W0)
        lodetP0=fastLogDet(psi0,W0)
        
        N=4
        X=np.random.multivariate_normal(np.zeros([d,]),LA.inv(Cov0),N) 
        Y0=np.ones((int(N/2),1))*0
        Y1=np.ones((int(N/2),1))*1
        y=np.concatenate((Y0,Y1))
    
        theta=np.random.uniform(size=[d,1])
        logPdf=bayesianlogisticPdf(theta,mu0,LA.inv(Cov0),lodetP0,X,y,Z=1)
        print("logPdf=",logPdf)
        logPdf_LS=bayesianlogisticPdf_largeScale(theta,mu0,W0,psi0,lodetP0,X,y,Z=1)
        print("logPdf_LS=",logPdf_LS)
        

        
        
        
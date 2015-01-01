# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.95
gamma = 2.
sigma = 1.
#sigma_e = np.array([0.04,0.05,0.1,0.05])
sigma_e = np.array([0.02])
sigma_E = np.diag(0.01*np.ones(1))
mu_e = 0.
mu_a = 0.
chi = 1.
delta = 0.06
xi_k = 0.33 #*.75



n = 2 # number of measurability constraints
nG = 2 # number of aggregate measurability constraints.
ny = 11 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 6 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 3 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
nZ = 2 # number of aggregate states
nEps = len(sigma_E) 

neps = len(sigma_e)

phat = np.array([-0.01])




def F(w):
    '''
    Individual first order conditions
    '''
    logm,nu,i,logc,logk_,f,fk,t1,xi_,a_,b_ = w[:ny] #y
    Ek_,Ea_,EUc_fk,EUc = w[ny:ny+ne] #e
    logK,Theta,logXi_,R_,T,T1 = w[ny+ne:ny+ne+nY] #Y
    logm_,nu_,i_= w[ny+ne+nY:ny+ne+nY+nz] #z
    xi,a,b = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    rho = w[ny+ne+nY+nz+nv] #v
    logK_,Theta_ = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    eps = w[ny+ne+nY+nz+nv+n_p+nZ] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps:ny+ne+nY+nz+nv+n_p+nZ+neps+nEps] #aggregate shock

    try:
        Theta_i = Eps[int(ad.value(i))]    
    except:
        Theta_i = Eps[int(i)]
            
    Xi_ = np.exp(logXi_)
    m_,m = np.exp(logm_),np.exp(logm)
    c,k_ = np.exp(logc),np.exp(logk_)
    
    Uc = c**(-sigma)
    A = np.exp(nu+Theta)
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = k_ - Ek_
    ret[1] = a_ - Ea_
    
    ret[2] = nu - rho*nu_+Theta_i
    ret[3] = i- i_
    ret[4] = xi - m * Uc
    ret[5] = xi_ - Xi_
    ret[6] = f - (A*k_**xi_k + (1-delta)*k_)
    ret[7] = fk - ( A * xi_k * k_**(xi_k-1) + (1-delta) )
    ret[8] = R_*(a_-k_) + f - c - a
    ret[9] = t1 - b
    ret[10] = b_ - (a_-k_)
    
    ret[11] = Xi_ - beta*m_*EUc_fk
    ret[12] = Xi_ - beta*m_*R_*EUc
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,nu,i,logc,logk_,f,fk,t1,xi_,a_,b_  = w[:ny] #y
    logK,Theta,logXi_,R_,T,T1 = w[ny+ne:ny+ne+nY] #Y
    logm_,nu_,i_= w[ny+ne+nY:ny+ne+nY+nz] #z
    logK_,Theta_ = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+nZ] #Z
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps:ny+ne+nY+nz+nv+n_p+nZ+neps+nEps] #aggregate shock
    
    c,k_ = np.exp(logc),np.exp(logk_)
    K_,K = np.exp(logK_),np.exp(logK)
    
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = logXi_
    ret[1] = R_
    
    ret[2] = logm
    ret[3] = T - (f-c-K)#(a_-k_)
    ret[4] = T1 - t1
    ret[5] = Theta - 0.6*Theta_
    
    ret[6] = K_ - k_
    ret[7] = a_-k_#f-c-K#implies f-c-K
    
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,nu,i,logc,logk_,f,fk,t1,xi_,a_,b_  = y #y
    
    c,k_ = np.exp(logc),np.exp(logk_)
    Uc = c**(-sigma)
    
            
    #Ek_,Ea_,EUc_fk,EUc 
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = k_
    ret[1] = a_
    ret[2] = Uc * fk
    ret[3] = Uc
        
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,nu,i = z
    logK,Theta,logXi,R,T,T1  = YSS
    
    Xi = np.exp(logXi)
        
    
    m = np.exp(logm)
    
    A = np.exp(nu)
    
    Uc =Xi/m
    c = (Uc)**(-1/sigma)
    
    k_ = ((1/beta + delta - 1)/(A*xi_k))**(1/(xi_k-1))
    
    f = A*k_**xi_k + (1-delta)*k_
    fk = A*xi_k* k_**(xi_k-1) + 1-delta
    
    xi_ = Xi*np.ones(c.shape)
    
    b = -(f-c -k_)/(1/beta - 1)
    a = b+k_

    
    #logm,nu_a,nu_e,logc,logl,lognl,w_e,r,pi,f,b_,labor_res,res,x_,logk_    
    return np.vstack((
    logm,nu,i,np.log(c),np.log(k_),f,fk,a,xi_,a,b
    ))
    
    

    
def GSS(YSS,y_i,weights):
    '''
    Aggregate conditions for the steady state
    '''
    logm,nu,i,logc,logk_ ,f,fk,t1,xi_,a_,b_ = y_i
    logK,Theta,logXi,R,T,T1  = YSS
  
    c,k_ = np.exp(logc),np.exp(logk_)
    K = np.exp(logK)      
    
    return np.hstack([
    weights.dot(K - k_), weights.dot(f-c-k_),R-1/beta,T,T1,Theta
     ])
    
    
    
def nomalize(Gamma,weights =None):
    '''
    Normalizes the distriubtion of states if need be
    '''
    if weights == None:            
        Gamma[:,0] -= np.mean(Gamma[:,0])
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
    return Gamma
    
def check_extreme(z_i):
    '''
    Checks for extreme positions in the state space
    '''
    return False
    extreme = False
    if z_i[0] < -3. or z_i[0] > 6.:
        extreme = True
    if z_i[1] > 8. or z_i[1] < -5.:
        extreme = True
    return extreme
    
def check_SS(YSS):
    if YSS[0] < -5.:
        return False
    return True
    
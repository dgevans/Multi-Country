# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:14:02 2014

@author: dgevans
"""

import approximate_aggstate_noshock as approximate
import numpy as np
from IPython.parallel import Client
from IPython.parallel import Reference
c = Client()
v = c[:] 

def simulate(Para,Gamma,Y,Shocks,y,T,T0=0,quadratic = True):
    '''
    Simulates a sequence of state path for a given Para
    '''
    approximate.calibrate(Para)
    t = T0+1
    while t< T:
        print t
        Gamma[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel(Para,Gamma[t-1],quadratic)
        t += 1
    return Gamma,Y,Shocks,y


def simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks,quadratic = True):
    '''
    Runs a simulation for a prespecified seqence of shocks
    '''
    approximate.calibrate(Para)
    Gamma,Y,Shocks,y = {},{},{},{}
    Gamma[0] = Gamma0 #initialize 100 agents at m = 1 for testing purposes    
    T=len(agg_shocks)
    t = 1
    while t< T:
        
        v['agg_shock']=agg_shocks[t]
        print t
        v.execute('approximate.shock=agg_shock')

        Gamma[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel(Para,Gamma[t-1],quadratic)
        t += 1
    return Gamma,Y,Shocks,y

    
def update_state_parallel(Para,Gamma,quadratic = True):
    '''
    Updates the state using parallel code
    '''
    v.block = True
    v['Gamma'] = Gamma
    v.execute('approx = approximate.approximate(Gamma)')
    approx = Reference('approx')
    diff = np.inf
    n = 0.
    while diff > 0.001 and n < 1:
        Gamma_new_t,Y_t,Shocks_t,y_t = filter(None,v.apply(lambda approx,quadratic = quadratic: approx.iterate(quadratic),approx))[0]
        error = np.linalg.norm(np.mean(Gamma_new_t[:,:2],0))
        if error < diff:
            diff = error
            Gamma_new,Y,Shocks,y = Gamma_new_t,Y_t,Shocks_t,y_t
        n += 1
    return Para.nomalize(Gamma_new.copy()),Y.copy(),Shocks.copy(),y.copy()

def update_state(Para,Gamma):
    '''
    Updates the state given current state gamma
    '''    
    eps = approximate.eps
    Izy = approximate.Izy
    sigma = Para.sigma_e
    N = len(Gamma)
    
    v['Gamma'] = Gamma
    v.execute('approx = approximate.approximate(Gamma)',block=True)
    v.execute('approximation = approx.get_approximation()',block=True)
    approx = Reference('approx')
    #dy,d2y = approx.dy,approx.d2y
    epsilon = np.random.randn(len(Gamma)) * sigma        
    #for z_i,z_i_new,e,y_i in zip(Gamma,Gamma_new,epsilon,y): 
    #    y_i[:] = approx.ss.get_y(z_i).flatten() + dy[eps](z_i).flatten()*e + 0.5*(d2y[eps,eps](z_i).flatten()*e**2 + d2y[sigma](z_i).flatten()*sigma**2).flatten()
    def compute_y(approx,eps,sigma,z_i,e):
        return approx.ss.get_y(z_i).flatten() + approx.dy[eps](z_i).flatten()*e + 0.5*(approx.d2y[eps,eps](z_i).flatten()*e**2 + approx.d2y[sigma](z_i).flatten()*sigma**2).flatten()
    
    y = np.vstack(v.map(compute_y,
               [approx]*N,[eps]*N,[sigma]*N,Gamma,epsilon,block=True))
        
    Gamma_new = Izy.dot(y.T).T
    
    approx = c[0]['approximation']
    d2Y = approx.d2Y
    Y = approx.ss.get_Y() + 0.5*d2Y[sigma].flatten()*sigma**2
    return Para.nomalize(Gamma_new),Y,epsilon,y
    
    
def simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T,T0=0,quadratic = True):
    '''
    Simulates a sequence of state path for a given Para
    '''
    #approximate.calibrate(Para)
    #v.execute('approxes = []')
    t = T0+1
    while t< T:
        Gamma[t],Z[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel_aggstate(Para,Gamma[t-1],Z[t-1],quadratic)
        print t,np.exp(Z[t]),Y[t-1][5:7]
        t += 1
    
def update_state_parallel_aggstate(Para,Gamma,Z,quadratic = True):
    '''
    Updates the state using parallel code
    '''
    v.block = True
    v['Gamma'] = Gamma
    v['Z'] = Z
    v.execute('approx = approximate.approximate(Gamma)')
    #v.execute('approxes.append(approx)')
    diff = np.inf
    n = 0.
    while diff > 0.001 and n < 1:
        v.execute('data = approx.iterate(Z)')
        Gamma_new_t,ZNew_t,Y_t,Shocks_t,y_t = filter(None,v['data'])[0]
        error = np.linalg.norm(np.mean(Gamma_new_t[:,:2],0))
        Gamma_new,ZNew,Y,Shocks,y = Gamma_new_t,ZNew_t,Y_t,Shocks_t,y_t
        n += 1
    return Para.nomalize(Gamma_new.copy()),ZNew.copy(),Y.copy(),Shocks.copy(),y.copy()
    
    
def simulate_aggstate_MPI(Para,Gamma,Z,Y,Shocks,y,T,T0=0,quadratic = True):
    '''
    Simulates using the MPI code rather than Ipython parallel
    '''
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:14:02 2014

@author: dgevans
"""

approximate = None
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T,T0=0,agg_shocks=None,quadratic = True,weights =None):
    '''
    Simulates using the MPI code rather than Ipython parallel
    '''
    approximate.calibrate(Para)
    approximate.shock = None
    t = T0+1
    while t < T:
        if agg_shocks is not None:
            approximate.shock = agg_shocks[t-1]
        Gamma[t],Z[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel_aggstate(Para,Gamma[t-1],Z[t-1],quadratic,weights)
        comm.Bcast([Gamma[t],MPI.DOUBLE])
        comm.Bcast([Z[t],MPI.DOUBLE])
        if rank ==0:
            print t,np.exp(Z[t]),Y[t-1][5:7]
        t += 1
        
    
def update_state_parallel_aggstate(Para,Gamma,Z,quadratic = True,weights=None):
    '''
    Updates the state using parallel code
    '''

    approx = approximate.approximate(Gamma,Gamma_weights=weights)
    data = approx.iterate(Z)
    if rank == 0:
        Gamma_new,ZNew,Y,Shocks,y = data
        return Para.nomalize(Gamma_new,weights),ZNew,Y,Shocks,y
    else:
        return np.empty(Gamma.shape),np.empty(Z.shape),None,None,None
        
        
def simulate_aggstate_ConditionalMean(Para,Gamma,Z,Y,Shocks,y,T,T0=0,quadratic = True,weights=None):
    '''
    Simulates using the MPI code rather than Ipython parallel
    '''
    #approximate.calibrate(Para)
    t = T0+1
    while t < T:
        Gamma[t],Z[t],Y[t-1], Shocks[t-1],y[t-1]= update_state_parallel_aggstate_ConditionalMean(Para,Gamma[t-1],Z[t-1],quadratic,weights)
        comm.Bcast([Gamma[t],MPI.DOUBLE])
        comm.Bcast([Z[t],MPI.DOUBLE])
        if rank ==0:
            print t,np.exp(Z[t]),Y[t-1][5:7]
        t += 1
        
    
def update_state_parallel_aggstate_ConditionalMean(Para,Gamma,Z,quadratic = True,weights=None):
    '''
    Updates the state using parallel code
    '''

    approx = approximate.approximate(Gamma,Gamma_weights=weights)
    data = approx.iterate_ConditionalMean(Z)
    if rank == 0:
        Gamma_new,ZNew,Y,Shocks,y = data
        return Para.nomalize(Gamma_new,weights),ZNew,Y,Shocks,y
    else:
        return np.empty(Gamma.shape),np.empty(Z.shape),None,None,None
    
    

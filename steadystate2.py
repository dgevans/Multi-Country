# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:30:59 2014

@author: dgevans
"""
import numpy as np
from scipy.optimize import root
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


Finv = None
GSS = None
check_SS = None


ny = None
ne = None
nY = None
nz = None
nv = None

Y0 = None

def calibrate(Para):
    global Finv,GSS,ny,ne,nY,nz,nv,Y0,check_SS
    Finv,GSS,check_SS = Para.Finv,Para.GSS,Para.check_SS
    ny,ne,nY,nz,nv = Para.ny,Para.ne,Para.nY,Para.nz,Para.nv
    Y0 =0.5*np.ones(nY)
    
class steadystate(object):
    '''
    Computes the steady state
    '''
    def __init__(self,dist,solve = True):
        '''
        Solves for the steady state given a distribution z_i
        '''
        self.Gamma = zip(*dist)
        self.Gamma[0],self.Gamma[1] = np.vstack(self.Gamma[0]),np.hstack(self.Gamma[1]) 
        if solve:
            self.solveSteadyState()
        
    def solveSteadyState(self):
        '''
        Solve for the steady state
        '''
        global Y0
        res = root(self.SteadyStateRes,Y0,tol = 1e-14)
        state = np.random.get_state()
        while not res.success or not check_SS(res.x):
            res = root(self.SteadyStateRes,np.random.rand(nY),tol=1e-14)
        np.random.set_state(state)
        
        diff = np.max(np.abs(self.SteadyStateRes(res.x)))
        diffs =  comm.allgather(diff)       
        i = np.argmin(diffs)
        
        Y0 = res.x
        comm.Bcast([Y0, MPI.DOUBLE],root=i)
        
        self.Y = Y0
        
    def SteadyStateRes(self,Y):
        '''
        For a given vector of aggregates returns the steady state residual
        '''
        y_i = Finv(Y,self.Gamma[0].T)
        return GSS(Y,y_i,self.Gamma[1])
        
    def get_Y(self):
        '''
        Gets the aggregate variables for the steady state
        '''
        return self.Y
        
    def get_y(self,z):
        '''
        Given idiosyncratic state z returns the idiosyncratic steady state values
        y
        '''
        return Finv(self.Y,z)
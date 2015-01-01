# -*- coding: utf-8 -*-
import steadystate
import calibrations.calibrate_complete as Para
import approximate
import numpy as np
import simulate_MPI as simulate
import utilities
from mpi4py import MPI
import cPickle
import warnings
warnings.filterwarnings('ignore')

simulate.approximate = approximate
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = {}

T = 300
approximate.Ntest = 10000
Para.k = 200
for N in [1,4,8,16,32,96]:
    Para.nEps = N
    Para.sigma_E = 0.01 * np.eye(N)
    approximate.calibrate(Para)
    
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    Gamma[0] = np.zeros((N,3))
    Gamma[0][:,2] = np.arange(N)    
    
    steadystate.calibrate(Para)
    ss = steadystate.steadystate(zip(np.zeros((1,3)),np.ones(1)))
    Z[0] = ss.get_Y()[:1]
    
    simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
    
    resids = {}
    for t in range(1,T,10):
        if rank == 0:
            print t
        approx = approximate.approximate(Gamma[t])
        resids[t] = approx.CheckEuler(y[t-1],Z[t])
    if rank == 0:
        data[N] = Gamma,Z,Y,Shocks,y,resids
    
if rank == 0:
    fout = file('complete_simulation.dat','wr')
    cPickle.dump(data,fout)
    fout.close()
    
    

        
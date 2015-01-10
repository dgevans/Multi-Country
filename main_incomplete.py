# -*- coding: utf-8 -*-
import steadystate
import calibrations.calibrate_incomplete2 as Para
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

T = 1000
approximate.Ntest = 20000
Para.k = 200
for N in [1,4,8,16,32,96]:
    if rank == 0:
        utilities.sendMessage('Incomplete Markets','Starting: ' + str(N) )
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
    for t in range(1,T,50):
        approx = approximate.approximate(Gamma[t])
        if rank == 0:
            print t
            resids[t] = approx.CheckEuler(y[t-1],Z[t]).mean(0)
    if rank == 0:
        data[N] = Gamma,Z,Y,Shocks,y,resids
        fout = file('incomplete_simulation.dat','wr')
        cPickle.dump(data,fout)
        fout.close()
        utilities.sendMessage('Incomplete Markets', 'Finished: ' + str(N) )
    

        
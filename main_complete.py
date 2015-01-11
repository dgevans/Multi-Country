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

if rank == 0:
    np.random.seed(5876954603456456345)

data = {}

T = 1000
approximate.Ntest = 20000
Para.k = 200
for N in [1,4,8,16,32,96]:
    if rank == 0:
        utilities.sendMessage('Complete Markets','Starting: ' + str(N) )
    Para.nEps = N
    Para.sigma_E = 0.01 * np.eye(N)
    Para.sigma_vec = 1. + 0.2*np.random.randn(N)
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
        temp = approx.CheckEuler(y[t-1],Z[t])
        if rank == 0:
            print t
            resids[t] = temp.mean(0)
    if rank == 0:
        data[N] = Gamma,Z,Y,Shocks,y,resids
        fout = file('complete_simulation_hetero.dat','wr')
        cPickle.dump(data,fout)
        fout.close()
        utilities.sendMessage('Complete Markets', 'Finished: ' + str(N) )

    
    
    

        
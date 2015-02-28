# -*- coding: utf-8 -*-
import steadystate
import calibrations.calibrate_incomplete_labor as Para
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

if rank == 0:
    np.random.seed(3244390)

T = 1000
approximate.Ntest = 25000
approximate.degenerate = True
Para.k = 200
for N in [1,4,8,16,32,96]:
    if rank == 0:
        utilities.sendMessage('Incomplete Markets','Starting: ' + str(N) )
    Para.nEps = N
    Para.sigma_E = 0.01 * np.eye(N)
    if rank ==0:
        Para.sigma_vec = 1. + 0.2*np.random.randn(N)
        Para.delta_vec = 0.02 + 0.002*np.random.randn(N)
    Para.sigma_vec = comm.bcast(Para.sigma_vec)# make sure sigmas are the same
    Para.delta_vec = comm.bcast(Para.delta_vec)
    
    approximate.calibrate(Para)
    
    Gamma,Z,Y,Shocks,y = {},{},{},{},{}
    Gamma[0] = np.zeros((N,3))
    Gamma[0][:,2] = np.arange(N)    
    approx = approximate.approximate(Gamma[0],fit=False)
    Z[0] = approx.ss.get_Y()[:1]
    
    simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,T)
    
    resids = {}
    for t in range(1,T,50):
        approx = approximate.approximate(Gamma[t])
        temp = approx.CheckEuler(y[t-1],Z[t])
        if rank == 0:
            print t
            resids[t] = temp.mean(0)
    if rank == 0:
        Names = 'Gamma','Z','Y','Shocks','y','resids','sigma','delta'
        data[N] = dict(zip(Names,[Gamma,Z,Y,Shocks,y,resids,Para.sigma_vec,Para.delta_vec]))
        fout = file('incomplete_simulation_labor_hetero_deg.dat','wr')
        cPickle.dump(data,fout)
        fout.close()
        utilities.sendMessage('Incomplete Markets', 'Finished: ' + str(N) )
    

        
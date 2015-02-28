# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:40:43 2015

@author: dgevans
"""
import numpy as np
import pandas as pd
import cPickle

data_cm = cPickle.load(file('complete_simulation_labor.dat','r'))
data_im = cPickle.load(file('incomplete_simulation_labor.dat','r'))
data_cm_deg = cPickle.load(file('complete_simulation_labor_deg.dat','r'))
data_im_deg = cPickle.load(file('incomplete_simulation_labor_deg.dat','r'))

Ns = np.sort(data_cm.keys())

table1 = pd.DataFrame(data={'N':Ns})

max_error = []
ave_error = []
res_error = []
max_error_deg = []
ave_error_deg = []

def get_resource_error(n,t):
    logm,nu,i,logc,logk_,logl,logf,logfk,xi_ = data_cm[n][4][t].T
    logK,logXi_ =  data_cm[n][2][t]
    
    return np.mean(np.exp(logf)-np.exp(logc) - np.exp(logK))/np.mean(f)

for n in Ns:
    max_error.append(np.log10(np.amax(np.abs(np.vstack(data_cm[n][5].values())))))
    ave_error.append(np.log10(np.abs(np.mean(np.vstack(data_cm[n][5].values())))))
    max_error_deg.append(np.log10(np.amax(np.abs(np.vstack(data_cm_deg[n][5].values())))))
    ave_error_deg.append(np.log10(np.abs(np.mean(np.vstack(data_cm_deg[n][5].values())))))
    #res_error.append(np.log10(np.amax(np.abs(
    #map(lambda t:get_resource_error(n,t), np.arange(999) )
    #))))
    
    
table1['Maximum Error'] = max_error
table1['Average Error'] = ave_error
#table1['Resource Error'] = res_error
table1['Maximum Error Degenerate'] = max_error_deg
table1['Average Error Degenerate'] = ave_error_deg
table1 = table1.set_index('N')


table2 = pd.DataFrame(data={'N':Ns})

max_error = []
ave_error = []
res_error = []
max_error_deg = []
ave_error_deg = []

def get_resource_error_im(n,t):
    logm,nu,i,logc,logk_,logf,logfk,xi_,a_  = data_im[n][-2][t].T
    logK,logXi_,R_,T =  data_im[n][-4][t]
    f  = np.exp(logf)
    return np.mean(f-np.exp(logc) - np.exp(logK))/np.mean(f)

for n in Ns:
    max_error.append(np.log10(np.amax(np.abs(np.vstack(data_im[n][5].values())))))
    ave_error.append(np.log10(np.abs(np.mean(np.vstack(data_im[n][5].values())))))
    max_error_deg.append(np.log10(np.amax(np.abs(np.vstack(data_im_deg[n][5].values())))))
    ave_error_deg.append(np.log10(np.abs(np.mean(np.vstack(data_im_deg[n][5].values())))))
    #res_error.append(np.log10(np.amax(np.abs(
    #map(lambda t:get_resource_error_im(n,t), np.arange(999) )
    #))))
    
    
table2['Maximum Error'] = max_error
table2['Average Error'] = ave_error
#table1['Resource Error'] = res_error
table2['Maximum Error Degenerate'] = max_error_deg
table2['Average Error Degenerate'] = ave_error_deg
table2 = table2.set_index('N')

#Now heterogeneous agents
data_cm = cPickle.load(file('complete_simulation_hetero.dat','r'))
data_im = cPickle.load(file('incomplete_simulation_hetero.dat','r'))

Ns = np.sort(data_cm.keys())

table3 = pd.DataFrame(data={'N':Ns})

max_error = []
ave_error = []
res_error = []

def get_resource_error_hetero(n,t):
    logm,nu,i,logc,logk_,logl,logf,logfk,xi_ = data_cm[n][-3][t].T
    logK,logXi_ =  data_cm[n][-5][t]
    return np.mean(np.exp(logf)-np.exp(logc) - np.exp(logK))/np.mean(np.exp(logf))

for n in Ns:
    max_error.append(np.log10(np.amax(np.abs(np.vstack(data_cm[n][-2].values())))))
    ave_error.append(np.log10(np.abs(np.mean(np.vstack(data_cm[n][-2].values())))))
    res_error.append(np.log10(np.amax(np.abs(
    map(lambda t:get_resource_error_hetero(n,t), np.arange(500,999) )
    ))))
    
    
table3['Maximum Error'] = max_error
table3['Average Error'] = ave_error
table3['Resource Error'] = res_error
table3 = table3.set_index('N')


table4 = pd.DataFrame(data={'N':Ns})

max_error = []
ave_error = []
res_error = []

def get_resource_error_hetero_im(n,t):
    logm,nu,i,logc,logk_,logl,logf,logfk,xi_,a_  = data_im[n][-3][t].T
    logK,logXi_,R_,T =  data_im[n][-5][t]
    f  = np.exp(logf)
    return np.mean(f-np.exp(logc) - np.exp(logK))/np.mean(f)

for n in Ns:
    max_error.append(np.log10(np.amax(np.abs(np.vstack(data_im[n][-2].values())))))
    ave_error.append(np.log10(np.abs(np.mean(np.vstack(data_im[n][-2].values())))))
    res_error.append(np.log10(np.amax(np.abs(
    map(lambda t:get_resource_error_hetero_im(n,t), np.arange(500,999) )
    ))))
    
    
table4['Maximum Error'] = max_error
table4['Average Error'] = ave_error
table4['Resource Error'] = res_error
table4 = table4.set_index('N')

import calibrations.calibrate_complete_agg as Para
import approximate
Para.k = 200

def time_to_compute(N):
    Para.nEps = 1 
    Para.sigma_E = 0.01 *np.eye(1) 
    Para.sigma_vec = 1. + 0.2*np.random.randn(N)
    Para.delta_vec = 0.02 + 0.002*np.random.randn(N)
    approximate.calibrate(Para)
    
    Gamma = np.zeros((N,3))
    Gamma[:,2] = np.arange(N)
    approx = approximate.approximate(Gamma)
    approx.iterate(approx.ss.get_Y()[:1])

import time
times = []
for N in np.arange(1,100,5):
    start = time.clock()
    time_to_compute(N)
    end = time.clock()
    times.append(end-start)
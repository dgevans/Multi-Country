# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:26:22 2014

@author: dgevans
"""

import numpy as np
import pycppad as ad
from mpi4py import MPI
import itertools
import smtplib
from email.mime.text import MIMEText

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def dict_map(F,l):
    '''
    perform a map preserving the dict structure
    '''
    ret = {}
    temp = map(F,l)
    keys = temp[0].keys()
    for key in keys:
        ret[key] = [t[key] for t in temp]
    return ret
    
    
    
def ad_Jacobian(F,x0):
    '''
    Computes Jacobian using automatic differentiation
    '''
    a_x = ad.independent(x0)
    a_F = F(a_x)
    return ad.adfun(a_x,a_F).jacobian(x0)

def ad_Hessian(F,x0):
    '''
    Computes Hessian of F using automatic differentiation
    '''
    a_x = ad.independent(x0)
    a_F = F(a_x)
    n = x0.shape[0]
    m = a_F.shape[0]
    HF = np.empty((m,n,n))
    I = np.eye(m)
    
    adF = ad.adfun(a_x,a_F)
    for i in range(m):
        HF[i,:,:] = adF.hessian(x0,I[i])
    
    return HF

from hashlib import sha1

from numpy import all, array


class hashable_array(np.ndarray):
    __hash = None
    def __new__(cls, values):
        this = np.ndarray.__new__(cls, shape=values.shape, dtype=values.dtype)
        this[...] = values
        return this
    
    def __init__(self, values):
        self.__hash = int(sha1(self).hexdigest(), 16)
    
    def __eq__(self, other):
        return all(np.ndarray.__eq__(self, other))
    
    def __hash__(self):
        if self.__hash == None:
            self.__hash = int(sha1(self).hexdigest(), 16)
        return self.__hash
    
    def __setitem__(self, key, value):
        raise Exception('hashable arrays are read-only')
        
def quadratic_dot(Q,a,b):
    '''
    Performs to the dot product appropriately
    '''
    return np.tensordot(np.tensordot(Q,a,(1,0)),b,(1,0))
    
def quadratic_solve(Ds,A,B,C):
    '''
    Solves the equations A*Q(D1,D2) + B * Q + C = 0.  Ds = [D1,D2] where D1 and D2 are 
    diagonal matrices (assumed to be just the diagonals).
    '''
    D1,D2 = Ds
    if C.shape[-2:] == (len(D1),len(D2)):
        Q = np.empty(C.shape)
        for i,rho1 in enumerate(D1):
            for j,rho2 in enumerate(D2):
                temp = np.linalg.inv(rho1*rho2*A + B)
                Q[:,i,j] = np.einsum('kl...,l...->k...',temp,-C[...,i,j])
        return Q
    else:
        Q = np.empty(np.hstack((C.shape,(len(D1),len(D2)))))
        for i,rho1 in enumerate(D1):
            for j,rho2 in enumerate(D2):
                temp = np.linalg.inv(rho1*rho2*A + B)
                Q[:,:,i,j] = np.einsum('kl...,l...->k...',temp,-C)
        return Q
        
def quadratic_solve2(A,B):
    '''
    Solves the quadratic equation A*Q+B = 0 where A*Q takes the form of
    np.einsum('...jkl,jkl->...kl',A,Q)
    '''
    nk,nl = A.shape[-2:]
    Q = np.empty(B.shape)
    for k in range(nk):
        for l in range(nl):
            temp = np.linalg.inv(A[:,:,k,l])
            Q[:,k,l] = np.einsum('ij...,j...->i...',temp,-B[:,k,l])
    return Q
    
def quadratic_solve3(D,A,B,C,induce = False):
    '''
    Solves the equations A*Q(.,D) + B * Q + C = 0.  where D is a diagonal matrix
    diagonal matrices (assumed to be just the diagonals).
    '''
    if not induce:
        Q = np.empty(C.shape)
        for i,rho in enumerate(D):
            temp = np.linalg.inv(rho*A + B)
            Q[:,:,i] = np.einsum('kl...,l...->k...',temp,-C[...,i])
        return Q
    else:
        Q = np.empty(np.hstack((C.shape,len(D))))
        for i,rho in enumerate(D):
            temp = np.linalg.inv(rho*A + B)
            Q[:,:,i] = np.einsum('kl...,l...->k...',temp,-C)
        return Q
    
    
class dict_fun(object):
    '''
    Creates a copy function which stores the results in a dictionary
    '''
    def __init__(self,f):
        '''
        Initialize with function f
        '''
        self.f = f
        self.fm = {}
    def __call__(self,z):
        '''
        Evaluates at a point z, first checks if z in in dictionary, if so, returns
        stored result.        
        '''
        #hash_z = sha1(np.ascontiguousarray(z)).hexdigest()
        hash_z = hash(z.tostring())
        if self.fm.has_key(hash_z):
            return self.fm[hash_z]
        else:
            f = self.f(z)
            self.fm[hash_z] = f
            return f
            
    def join(self,fast=False):
        '''
        Joins the dictionaries across multiple instances 
        '''
        if not fast:
            my_data = self.fm.items()
            data = comm.gather(my_data)
            data = comm.bcast(data)
            self.fm = dict(itertools.chain(*data))
        else:
            my_keys = np.hstack(self.fm.keys())
            n = len(my_keys)
            
            shape = self.fm.values()[0].shape
            my_values = np.ascontiguousarray(np.vstack(self.fm.values()).reshape(np.hstack((-1,shape))))
            
            keys = np.empty(size*n,int)
            values = np.empty(np.hstack((size*n,shape)))
            comm.Allgather([my_keys,MPI.INT],[keys,MPI.INT])
            comm.Allgather([my_values,MPI.DOUBLE],[values,MPI.DOUBLE])
            
            self.fm = dict(itertools.izip(keys,values))

def parallel_map(f,X):
    '''
    A map function that applies f to each element of X
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = range(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_data =  map(f,X[my_range])
    data = comm.gather(my_data)
    data = comm.bcast(data)
    return list(itertools.chain(*data))
    
def parallel_sum(f,X):
    '''
    In parallel applies f to each element of X and computes the sum
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = range(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_sum =  sum(itertools.imap(f,X[my_range]))
    sums = comm.gather(my_sum)
    return sum( comm.bcast(sums) )
    
def parallel_dict_map(F,l):
    '''
    perform a map preserving the dict structure
    '''
    ret = {}
    temp = parallel_map(F,l)
    keys = temp[0].keys()
    for key in keys:
        ret[key] = [t[key] for t in temp]
    return ret
    
def transpose_sum(A):
    '''
    Returns the sum of A and its transpose
    '''
    return A + A.transpose(0,2,1)
    
    
def sendMessage(subject,message):
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login('econcluster@gmail.com','password')
    msg = MIMEText(message)
    msg['To'] = 'dgevans42@gmail.com'
    msg['From'] = 'econcluster@gmail.com'
    msg['Subject'] = subject
    server.sendmail(msg['From'],msg['To'],msg.as_string())
    server.close()
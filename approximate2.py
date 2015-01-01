# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:28:52 2014

@author: dgevans
"""
import steadystate2 as steadystate
import numpy as np
import utilities
from utilities import hashable_array
from utilities import quadratic_dot
from utilities import quadratic_solve
from utilities import quadratic_solve2
from utilities import quadratic_solve3
from utilities import dict_fun
from utilities import transpose_sum
import itertools
from scipy.cluster.vq import kmeans2
from scipy.optimize import root
from scipy.linalg import solve_sylvester

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

timing = np.zeros(6) 

def mdot(*args):
    return reduce(np.dot, args)

def parallel_map(f,X):
    '''
    A map function that applies f to each element of X
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_data =  map(f,X[my_range])
    data = comm.gather(my_data)
    if rank == 0:
        return list(itertools.chain(*data))
    else:
        return None
        
def parallel_map_noret(f,X):
    '''
    A map function that applies f to each element of X
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    map(f,X[my_range])
    
def parallel_sum(f,X):
    '''
    In parallel applies f to each element of X and computes the sum
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_sum =  np.ascontiguousarray(sum(itertools.imap(f,X[my_range])))
    if r==0:#if evenly split do it fast
        shape = my_sum.shape
        ret = np.empty(shape)
        comm.Allreduce([my_sum,MPI.DOUBLE],[ret,MPI.DOUBLE])
        return ret
    else:
        sums = comm.gather(my_sum)
        if rank == 0:
            res = sum(sums)
            comm.bcast(res.shape)
        else:
            shape = comm.bcast(None)
            res = np.empty(shape)
        comm.Bcast([res,MPI.DOUBLE])
        return res
    
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
    
    
F =None
G = None
f = None

n = None
nG = None
ny = None
ne = None
nY = None
nz = None
nv = None
n_p = None
nZ = None
neps = None
nEps = None
Ivy = None
Izy = None
IZY = None
IZYhat = None
Para = None
dY_Z0 = None
k = None
degenerate = False

shock = None

logm_min = -np.inf
muhat_min = -np.inf

y,e,Y,z,v,eps,Eps,p,Z,S,sigma,sigma_E = [None]*12
#interpolate = utilities.interpolator_factory([3])

def calibrate(Parahat):
    global F,G,f,ny,ne,nY,nz,nv,n,Ivy,Izy,IZY,nG,n_p,nZ,neps,nEps
    global y,e,Y,z,v,eps,p,Z,S,sigma,Eps,sigma_E,Para,k
    Para = Parahat
    #global interpolate
    F,G,f = Para.F,Para.G,Para.f
    ny,ne,nY,nz,nv,n,nG,n_p,neps,nZ,nEps = Para.ny,Para.ne,Para.nY,Para.nz,Para.nv,Para.n,Para.nG,Para.n_p,Para.neps,Para.nZ,Para.nEps
    Ivy,Izy,IZY = np.zeros((nv,ny)),np.zeros((nz,ny)),np.zeros((nZ,nY)) # Ivy given a vector of y_{t+1} -> v_t and Izy picks out the state
    Ivy[:,-nv:] = np.eye(nv)
    Izy[:,:nz] = np.eye(nz)
    IZY[:,:nZ] = np.eye(nZ)
        
    
    #store the indexes of the various types of variables
    y = np.arange(ny).view(hashable_array)
    e = np.arange(ny,ny+ne).view(hashable_array)
    Y = np.arange(ny+ne,ny+ne+nY).view(hashable_array)
    z = np.arange(ny+ne+nY,ny+ne+nY+nz).view(hashable_array)
    v = np.arange(ny+ne+nY+nz,ny+ne+nY+nz+nv).view(hashable_array)
    p = np.arange(ny+ne+nY+nz+nv,ny+ne+nY+nz+nv+n_p).view(hashable_array)
    Z = np.arange(ny+ne+nY+nz+nv+n_p,ny+ne+nY+nz+nv+n_p+nZ).view(hashable_array)
    eps = np.arange(ny+ne+nY+nz+nv+n_p+nZ,ny+ne+nY+nz+nv+n_p+nZ+neps).view(hashable_array)
    Eps = np.arange(ny+ne+nY+nz+nv+n_p+nZ+neps,ny+ne+nY+nz+nv+n_p+nZ+neps+nEps).view(hashable_array)
    
    S = np.hstack((z,Y)).view(hashable_array)
    
    sigma = Para.sigma_e.view(hashable_array)
    
    sigma_E = Para.sigma_E.view(hashable_array)
    
    k = Para.k
    
    #interpolate = utilities.interpolator_factory([3]*nz) # cubic interpolation
    steadystate.calibrate(Para)

class approximate(object):
    '''
    Computes the second order approximation 
    '''
    def __init__(self,Gamma,Gamma_weights=None,fit = True):
        '''
        Approximate the equilibrium policies given z_i
        '''
        if Gamma_weights ==None:
            Gamma_weights = np.ones(len(Gamma))/len(Gamma)
        self.Gamma = Gamma
        self.Gamma_weights = Gamma_weights
        self.approximate_Gamma()
        self.ss = steadystate.steadystate(self.dist)
        
        self.simple = False
        if (self.Gamma_ss == self.Gamma).all():
            self.simple = True

        #precompute Jacobians and Hessians
        self.get_w = dict_fun(self.get_wf)
        self.DF = dict_fun(lambda z_i:utilities.ad_Jacobian(F,self.get_w(z_i)))
        self.HF = dict_fun(lambda z_i:utilities.ad_Hessian(F,self.get_w(z_i)))
        
        self.DG = dict_fun(lambda z_i:utilities.ad_Jacobian(G,self.get_w(z_i)))
        self.HG = dict_fun(lambda z_i:utilities.ad_Hessian(G,self.get_w(z_i)))
        
        self.df = dict_fun(lambda z_i:utilities.ad_Jacobian(f,self.get_w(z_i)[y]))
        self.Hf = dict_fun(lambda z_i:utilities.ad_Hessian(f,self.get_w(z_i)[y]))
        
        if fit:
            #linearize
            self.linearize()
            self.quadratic()
            if not self.simple:
                self.join_function()
        
    def approximate_Gamma(self):
        '''
        Approximate the Gamma distribution
        '''
        #if rank == 0:
        #    cluster,labels = kmeans2(self.Gamma,k,minit='points')
        #    cluster,labels = comm.bcast((cluster,labels))
        #else:
        #    cluster,labels = comm.bcast(None)
        if not degenerate:
            cluster,labels = kmeans2(self.Gamma,self.Gamma[:k,:],minit='matrix')
        else:
            cluster = np.zeros((1,self.Gamma.shape[1]))
            labels = np.zeros(len(self.Gamma),dtype=int)
        weights = (labels == np.arange(k).reshape(-1,1)).dot(self.Gamma_weights)
        #weights = (labels-np.arange(k).reshape(-1,1) ==0).sum(1)/float(len(self.Gamma))
        #Para.nomalize(cluster,weights)
        self.Gamma_ss = cluster[labels,:]
        mask = weights > 0
        cluster = cluster[mask]
        weights = weights[mask]
        
        
        self.dist = zip(cluster,weights)        
        

        
    def integrate(self,f):
        '''
        Integrates a function f over Gamma
        '''
        def f_int(x):
            z,w = x
            return w*f(z)
        return parallel_sum(f_int,self.dist)
    
    def get_wf(self,z_i):
        '''
        Gets w for particular z_i
        '''
        ybar = self.ss.get_y(z_i).flatten()
        Ybar = self.ss.get_Y()
        Zbar = Ybar[:nZ]
        ebar = f(ybar)
        
        return np.hstack((
        ybar,ebar,Ybar,ybar[:nz],Ivy.dot(ybar),np.ones(n_p),Zbar,np.zeros(neps),np.zeros(nEps)
        ))
        
    def compute_dy(self):
        '''
        Computes dy w.r.t z_i, eps, Y
        '''
        self.dy = {}
        
        def dy_z(z_i):
            DF = self.DF(z_i)
            df = self.df(z_i)
            DFi = DF[n:] # pick out the relevant equations from F. Forexample to compute dy_{z_i} we need to drop the first equation
            return np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df)+DFi[:,v].dot(Ivy),
                                    -DFi[:,z])
        self.dy[z] = dict_fun(dy_z)
        
        def dy_eps(z_i):   
            DF = self.DF(z_i)    
            DFi = DF[:-n,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy),
                                    -DFi[:,eps])
                                    
        self.dy[eps] = dict_fun(dy_eps)
        
        self.compute_dy_Z()
                
        def dy_Y(z_i):
            DF = self.DF(z_i)
            df = self.df(z_i)                    
            DFi = DF[n:,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy),
                                -DFi[:,Y] - DFi[:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat))
        self.dy[Y] = dict_fun(dy_Y)
        
        
    def dY_Z_residual(self,dY_Z):
        def dy_Z(z_i):
            DFi = self.DF(z_i)[n:,:]
            df = self.df(z_i)
            dZ_Z = IZY.dot(dY_Z)
            A = np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df),DFi[:,v].dot(Ivy))
            B = np.linalg.inv(dZ_Z)
            C = -np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df),DFi[:,Y].dot(dY_Z)+DFi[:,Z]).dot(B)
            return solve_sylvester(A,B,C)
        
        DG = lambda z_i : self.DG(z_i)[nG:,:]
        return self.integrate(lambda z_i : DG(z_i)[:,y].dot(dy_Z(z_i)) + DG(z_i)[:,Y].dot(dY_Z) + DG(z_i)[:,Z])
        
        
    def compute_dy_Z(self):
        '''
        Computes linearization w/ respecto aggregate state.
        '''
        global dY_Z0
        def f(dY_Z):
            return self.dY_Z_residual(dY_Z.reshape(nY,nZ)).flatten()
        
        while True:
            if rank == 0:
                if dY_Z0 == None:
                    dY_Z0 =np.random.randn(nY*nZ)
            else:
                dY_Z0 = np.empty(nY*nZ)
            comm.Bcast([dY_Z0,MPI.DOUBLE])
            try:
                res = root(f,dY_Z0)
                if res.success:
                    lamb =  np.linalg.eigvals(res.x.reshape(nY,nZ)[:nZ])
                    if np.max(np.abs(lamb)) <1 and all(np.isreal(lamb)) :
                        break
                dY_Z0 = None
            except:
                dY_Z0 = None
            
                
        dY_Z = res.x.reshape(nY,nZ)
        dY_Z0 = res.x
        #Now change the basis so that dZ_Z is diagonal
        global IZYhat
        D,V = np.linalg.eig(dY_Z[:nZ])
        self.dY_Z = dY_Z.dot(V)
        self.dZ_Z = np.diagflat(D)
        self.dZhat_Z = np.linalg.inv(V)
        IZYhat = np.linalg.solve(V,IZY)
        
        def dy_Z(z_i):
            DFi = self.DF(z_i)[n:,:]
            df = self.df(z_i)
            A = np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df),DFi[:,v].dot(Ivy))
            B = np.linalg.inv(self.dZ_Z)
            C = -np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df),DFi[:,Y].dot(self.dY_Z)+DFi[:,Z].dot(V)).dot(B)
            return solve_sylvester(A,B,C)
            
            
        self.dy[Z] = dict_fun(dy_Z)
        
            
        
            
    def linearize(self):
        '''
        Computes the linearization
        '''
        self.compute_dy()
        DG = lambda z_i : self.DG(z_i)[nG:,:] #account for measurability constraints
        
        def DGY_int(z_i):
            DGi = DG(z_i)
            return DGi[:,Y]+DGi[:,y].dot(self.dy[Y](z_i))
        
        self.DGYinv = np.linalg.inv(self.integrate(DGY_int))
        
        def dYf(z_i):
            DGi = DG(z_i)
            return self.DGYinv.dot(-DGi[:,z]-DGi[:,y].dot(self.dy[z](z_i)))
        
        self.dY = dict_fun(dYf)
        
        self.linearize_aggregate()
        
        self.linearize_parameter()
    
    def linearize_aggregate(self):
        '''
        Linearize with respect to aggregate shock.
        '''
        dy = {}
        
        def Fhat_y(z_i):
            DFi = self.DF(z_i)[:-n,:]
            return DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy)
            
        Fhat_inv = dict_fun(lambda z_i: -np.linalg.inv(Fhat_y(z_i))) 
        
        def dy_dYprime(z_i):
            DFi = self.DF(z_i)[:-n,:]
            return Fhat_inv(z_i).dot(DFi[:,v]).dot(Ivy).dot(self.dy[Y](z_i))
            
        dy_dYprime = dict_fun(dy_dYprime)
        
        temp = np.linalg.inv( np.eye(nY) - self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i))) )
        
        self.temp_matrix_Eps = np.eye(nY) - self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i)))
        
        dYprime = {}
        dYprime[Eps] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[:-n,Eps]))        
        )
        dYprime[Y] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[:-n,Y] + self.DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat)))        
        )
        
        
        dy[Eps] = dict_fun(
        lambda z_i : Fhat_inv(z_i).dot(self.DF(z_i)[:-n,Eps]) + dy_dYprime(z_i).dot(dYprime[Eps])        
        )
        
        dy[Y] = dict_fun(
        lambda z_i : Fhat_inv(z_i).dot(self.DF(z_i)[:-n,Y] + self.DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat)) + dy_dYprime(z_i).dot(dYprime[Y])         
        )
        
        #Now do derivatives w.r.t G
        DGi = dict_fun(lambda z_i : self.DG(z_i)[:-nG,:])
        
        DGhat_Y = self.integrate(
        lambda z_i : DGi(z_i)[:,Y] + DGi(z_i)[:,y].dot(dy[Y](z_i))        
        )
        
        self.dY_Eps = -np.linalg.solve(DGhat_Y,self.integrate(
        lambda z_i : DGi(z_i)[:,Eps] + DGi(z_i)[:,y].dot(dy[Eps](z_i))          
        ) )
        
        self.dy[Eps] = dict_fun(
        lambda z_i : dy[Eps](z_i) + dy[Y](z_i).dot(self.dY_Eps)
        )
        
    def linearize_parameter(self):
        '''
        Linearize with respect to aggregate shock.
        '''
        dy = {}
        
        def Fhat_p(z_i):
            DFi = self.DF(z_i)[n:,:]
            df = self.df(z_i)
            return DFi[:,y]+ DFi[:,e].dot(df) + DFi[:,v].dot(Ivy) + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy)
            
        Fhat_inv = dict_fun(lambda z_i: np.linalg.inv(Fhat_p(z_i))) 
        
        def dy_dYprime(z_i):
            DFi = self.DF(z_i)[n:,:]
            return Fhat_inv(z_i).dot(DFi[:,v]).dot(Ivy).dot(self.dy[Y](z_i))
            
        dy_dYprime = dict_fun(dy_dYprime)
        
        temp = -np.linalg.inv( np.eye(nY) + self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i))) )
        
        dYprime = {}
        dYprime[p] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[n:,p]))        
        )
        dYprime[Y] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot( self.DF(z_i)[n:,Y] + self.DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat)) )        
        )
        
        
        dy[p] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[n:,p]) - dy_dYprime(z_i).dot(dYprime[p])        
        )
        
        dy[Y] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[n:,Y] +  self.DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat)) - dy_dYprime(z_i).dot(dYprime[Y])         
        )
        
        #Now do derivatives w.r.t G
        DGi = dict_fun(lambda z_i : self.DG(z_i)[nG:,:])
        
        DGhatinv = np.linalg.inv(self.integrate(
        lambda z_i : DGi(z_i)[:,Y] + DGi(z_i)[:,y].dot(dy[Y](z_i))        
        ))
        
        self.dY_p = -DGhatinv.dot( self.integrate(
        lambda z_i : DGi(z_i)[:,p] + DGi(z_i)[:,y].dot(dy[p](z_i))          
        ) )
        
        self.dy[p] = dict_fun(
        lambda z_i : dy[p](z_i) + dy[Y](z_i).dot(self.dY_p)
        )
        
                              
    def get_df(self,z_i):
        '''
        Gets linear constributions
        '''
        dy = self.dy
        d = {}
        df = self.df(z_i)
        
        d[y,S],d[y,eps],d[y,Z],d[y,Eps],d[y,p] = np.hstack((dy[z](z_i),dy[Y](z_i))),dy[eps](z_i), dy[Z](z_i),dy[Eps](z_i),dy[p](z_i) # first order effect of S and eps on y
    
        d[e,S],d[e,eps],d[e,Z],d[e,Eps],d[e,p] = df.dot(d[y,S]), np.zeros((ne,1)),df.dot(d[y,Z]), np.zeros((ne,nEps)),df.dot(d[y,p]) # first order effect of S on e
    
        d[Y,S],d[Y,eps],d[Y,Z],d[Y,Eps],d[Y,p] = np.hstack(( np.zeros((nY,nz)), np.eye(nY) )),np.zeros((nY,neps)),self.dY_Z,self.dY_Eps,self.dY_p
        
        d[z,S],d[z,eps],d[z,Z],d[z,Eps],d[z,p] = np.hstack(( np.eye(nz), np.zeros((nz,nY)) )),np.zeros((nz,neps)),np.zeros((nz,nZ)),np.zeros((nz,nEps)),np.zeros((nz,n_p))
        
        d[v,S],d[v,eps],d[v,Z] = Ivy.dot(d[y,S]) + Ivy.dot(dy[Z](z_i)).dot(IZYhat).dot(d[Y,S]), Ivy.dot(dy[z](z_i)).dot(Izy).dot(dy[eps](z_i)), Ivy.dot(dy[Z](z_i)).dot(self.dZ_Z)
        d[v,Eps] = Ivy.dot(dy[z](z_i).dot(Izy).dot(dy[Eps](z_i)) + dy[Y](z_i).dot(self.dYGamma_Eps) + dy[Z](z_i).dot(IZYhat).dot(self.dY_Eps) )
        d[v,p] =  Ivy.dot( dy[z](z_i).dot(Izy).dot(dy[p](z_i)) + dy[Y](z_i).dot(self.dYGamma_p) + dy[Z](z_i).dot(IZYhat).dot(self.dY_p) )      
        
        d[eps,S],d[eps,eps],d[eps,Z],d[eps,Eps],d[eps,p] = np.zeros((neps,nz+nY)),np.eye(neps),np.zeros((neps,nZ)),np.zeros((neps,nEps)),np.zeros((neps,n_p))
        
        d[Eps,S],d[Eps,eps],d[Eps,Z],d[Eps,Eps],d[Eps,p] = np.zeros((nEps,nz+nY)),np.zeros((nEps,neps)),np.zeros((nEps,nZ)),np.eye(nEps),np.zeros((nEps,n_p))
        
        d[Z,Z],d[Z,S],d[Z,Eps],d[Z,eps],d[Z,p] = np.linalg.inv(self.dZhat_Z),np.zeros((nZ,nY+nz)),np.zeros((nZ,nEps)),np.zeros((nZ,neps)),np.zeros((nZ,n_p))
        
        d[p,Z],d[p,S],d[p,Eps],d[p,eps],d[p,p] = np.zeros((n_p,nZ)),np.zeros((n_p,nY+nz)),np.zeros((n_p,nEps)),np.zeros((n_p,neps)),np.eye(n_p)

        d[y,z] = d[y,S][:,:nz]
        
        
        

        return d
     
    def compute_HFhat(self):
        '''
        Constructs the HFhat functions
        '''
        self.HFhat = {}
        shock_hashes = [eps.__hash__(),Eps.__hash__()]
        for x1 in [S,eps,Z,Eps,p]:
            for x2 in [S,eps,Z,Eps,p]:
                
                #Construct a function for each pair x1,x2
                def HFhat_temp(z_i,x1=x1,x2=x2):
                    HF = self.HF(z_i)
                    d = self.get_d(z_i)
                    HFhat = 0.
                    for y1 in [y,e,Y,Z,z,v,eps,Eps,p]:
                        HFy1 = HF[:,y1,:]
                        for y2 in [y,e,Y,Z,z,v,eps,Eps,p]:
                            if x1.__hash__() in shock_hashes or x2.__hash__() in shock_hashes:
                                HFhat += quadratic_dot(HFy1[:-n,:,y2],d[y1,x1],d[y2,x2])
                            else:
                                HFhat += quadratic_dot(HFy1[n:,:,y2],d[y1,x1],d[y2,x2])
                    return HFhat
                    
                self.HFhat[x1,x2] = dict_fun(HFhat_temp)
                
    def compute_d2y_ZZ(self):
        '''
        Copute the second derivative of y with respect to Z
        '''
        D = np.diagonal(self.dZ_Z)
        DF = self.DF
        
        def dy_YZZ(z_i):
            DFi,df,dy_Z = DF(z_i)[n:],self.df(z_i),self.dy[Z](z_i)
            A = DFi[:,v].dot(Ivy)
            B = DFi[:,y] + DFi[:,e].dot(df)
            C = DFi[:,Y] + mdot(DFi[:,v],Ivy,dy_Z,IZYhat) #DFi[:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat)
            return quadratic_solve([D,D],A,B,C)
            
        dy_YZZ = dict_fun(dy_YZZ)
        
        def d2y_ZZ(z_i):
            DFi,df,Hf,dy_Z = DF(z_i)[n:],self.df(z_i),self.Hf(z_i),self.dy[Z](z_i)
            A = DFi[:,v].dot(Ivy)
            B = DFi[:,y] + DFi[:,e].dot(df)
            C = self.HFhat[Z,Z](z_i) +np.einsum('ij...,j...->i...',DFi[:,e],quadratic_dot(Hf,dy_Z,dy_Z))
            return quadratic_solve([D,D],A,B,C)
            
        d2y_ZZ = dict_fun(d2y_ZZ)
        
        
        def HGhat(z_i,y1,y2):
            HG = self.HG(z_i)[nG:,:]
            d = self.get_d(z_i)
            HGhat = np.zeros((nY,len(y1),len(y2)))
            for x1 in [y,z,Y,Z]:
                HGx1 = HG[:,x1,:]
                for x2 in [y,z,Y,Z]:
                    HGhat += quadratic_dot(HGx1[:,:,x2],d[x1,y1],d[x2,y2])
            return HGhat
                    
        HGhat_ZZ = dict_fun(lambda z_i : HGhat(z_i,Z,Z))
        
        
        temp1 =self.integrate(lambda z_i : np.einsum('ij...,j...->i...',self.DG(z_i)[nG:,y],dy_YZZ(z_i)) + self.DG(z_i)[nG:,Y,np.newaxis,np.newaxis])
        temp2 = self.integrate(lambda z_i : np.einsum('ij,jkl',self.DG(z_i)[nG:,y],d2y_ZZ(z_i)) + HGhat_ZZ(z_i))

        
        self.d2Y[Z,Z] = quadratic_solve2(temp1,temp2)
        
        self.d2y[Z,Z] = dict_fun(lambda z_i: d2y_ZZ(z_i) + np.einsum('ijkl,jkl->ikl',dy_YZZ(z_i),self.d2Y[Z,Z]) )
        
        #d2y_ZS
        
        def d2y_SZ(z_i):
            DFi,df,Hf = DF(z_i)[n:],self.df(z_i),self.Hf(z_i)
            d,d2y_ZZ = self.get_d(z_i),self.d2y[Z,Z](z_i)
            temp = -(self.HFhat[S,Z](z_i) + np.tensordot(DFi[:,e],quadratic_dot(Hf,d[y,S],d[y,Z]),1)
                    + np.tensordot(DFi[:,v].dot(Ivy), quadratic_dot(d2y_ZZ,IZYhat.dot(d[Y,S]),self.dZ_Z),1) )
                    
            d2y_SZ = np.empty((ny,nY+nz,nZ))
            for i,rho in enumerate(D):
                 temp2 = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + rho*DFi[:,v].dot(Ivy))
                 d2y_SZ[:,:,i] = temp2.dot(temp[:,:,i])
            return d2y_SZ
            
        d2y_SZ = dict_fun(d2y_SZ)
        
        def dy_YSZ(z_i):
            DFi,df = DF(z_i)[n:],self.df(z_i)
            temp = DFi[:,Y] + DFi[:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat)
            dy_YSZ = np.empty((ny,nY,nZ))
            for i,rho in enumerate(D):
                temp2 = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + rho*DFi[:,v].dot(Ivy))
                dy_YSZ[:,:,i] = - temp2.dot(temp)
            return dy_YSZ
            
        dy_YSZ = dict_fun(dy_YSZ)
        
        HGhat_SZ = dict_fun(lambda z_i : HGhat(z_i,S,Z))
        HGhat_YZ = self.integrate(lambda z_i : (HGhat_SZ(z_i) + np.einsum('ij...,j...->i...',self.DG(z_i)[nG:,y],d2y_SZ(z_i)))[:,nz:,:])
        HGhat_zZ = lambda z_i : (HGhat_SZ(z_i) + np.einsum('ij...,j...->i...',self.DG(z_i)[nG:,y],d2y_SZ(z_i)))[:,:nz,:]
        
        Temp = self.integrate(lambda z_i : np.einsum('ij...,j...->i...',self.DG(z_i)[nG:,y],dy_YSZ(z_i)) + self.DG(z_i)[nG:,Y,np.newaxis])
        #Temp2 = self.integrate(lambda z_i : np.tensordot(self.DG(z_i)[nG:,y],d2y_SZ(z_i),1) + HGhat_SZ(z_i))
        
        def d2Y_zZ(z_i):
            temp2 = HGhat_zZ(z_i) + quadratic_dot(HGhat_YZ,self.dY(z_i),np.eye(nZ))
            d2Y_zZ = np.empty((nY,nz,nZ))
            for i in range(nZ):
                d2Y_zZ[:,:,i] = -np.einsum('ij...,j...->i...',np.linalg.inv(Temp[:,:,i]),temp2[:,:,i])
            return d2Y_zZ
            
        self.d2Y[z,Z] = dict_fun(d2Y_zZ)
        self.d2Y[Z,z] = lambda z_i : self.d2Y[z,Z](z_i).transpose(0,2,1)
        self.d2y[S,Z] = d2y_SZ
        self.dy[Y,S,Z] = dy_YSZ
        self.d2y[Z,S] = lambda z_i : d2y_SZ(z_i).transpose(0,2,1)
        
        def d2y_Zeps(z_i):
            DFi = DF(z_i)[:-n]
            temp = np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy))
            return -np.tensordot(temp, self.HFhat[Z,eps](z_i)
                                       + np.tensordot(DFi[:,v].dot(Ivy), 
                                       quadratic_dot(self.d2y[Z,S](z_i)[:,:,:nz],self.dZ_Z,Izy.dot(self.dy[eps](z_i))) ,1),1)
        self.d2y[Z,eps] = dict_fun(d2y_Zeps)
        self.d2y[eps,Z] = lambda z_i : self.d2y[Z,eps].transpose(0,2,1)
        
    def compute_d2y_Eps(self):
        '''
        Computes the 2nd derivatives of y with respect to the aggregate shock.
        '''

        def HGhat(z_i,y1,y2):
            HG = self.HG(z_i)[:-nG,:]
            d = self.get_d(z_i)
            HGhat = np.zeros((nY,len(y1),len(y2)))
            for x1 in [y,z,Y,Z,Eps]:
                HGx1 = HG[:,x1,:]
                for x2 in [y,z,Y,Z,Eps]:
                    HGhat += quadratic_dot(HGx1[:,:,x2],d[x1,y1],d[x2,y2])
            return HGhat        
        
        DF = self.DF
        def dSprime_Eps(z_i):
            return np.vstack((Izy.dot(self.dy[Eps](z_i)),self.dYGamma_Eps))
        
        d2Yprime_EpsEps = self.integrate(lambda z_i : quadratic_dot(self.d2Y[z,z](z_i),Izy.dot(self.dy[Eps](z_i)),Izy.dot(self.dy[Eps](z_i)))
                                            +2*quadratic_dot(self.d2Y[z,Y](z_i),Izy.dot(self.dy[Eps](z_i)),self.dYGamma_Eps)) 
        d2YprimeGZ_EpsEps1 = self.integrate(lambda z_i: np.einsum('ijk,jl,km->ilkm',self.d2Y[z,Z](z_i),Izy.dot(self.dy[Eps](z_i)),IZYhat.dot(self.dY_Eps)))
        d2YprimeGZ_EpsEps2 = self.integrate(lambda z_i: np.einsum('ijk,jl,km->ilkm',self.d2Y[z,Z](z_i),Izy.dot(self.dy[Eps](z_i)),IZYhat.dot(self.dYGamma_Eps)))
        
        def DFhat_EpsEps(z_i):
            dSprime_Eps_i,dZprime_Eps = dSprime_Eps(z_i),IZYhat.dot(self.dY_Eps)
            DFi = DF(z_i)[:-n]
            return self.HFhat[Eps,Eps](z_i) + np.tensordot(DFi[:,v].dot(Ivy),
            quadratic_dot(self.d2y[S,S](z_i),dSprime_Eps_i,dSprime_Eps_i)
            +quadratic_dot(self.d2y[S,Z](z_i),dSprime_Eps_i,dZprime_Eps)
            +quadratic_dot(self.d2y[S,Z](z_i),dSprime_Eps_i,dZprime_Eps).transpose(0,2,1)
            +quadratic_dot(self.d2y[Z,Z](z_i),dZprime_Eps,dZprime_Eps)    
            +np.tensordot(self.dy[Y](z_i),d2Yprime_EpsEps,1)
            +np.einsum('ijk,jlkm',self.dy[Y,S,Z](z_i),d2YprimeGZ_EpsEps1) #from dy_GammaZ
            +np.einsum('ijk,jlkm',self.dy[Y,S,Z](z_i),d2YprimeGZ_EpsEps1).transpose(0,2,1)
            +np.einsum('ijk,jlkm',self.d2y[Y,S,Z](z_i),d2YprimeGZ_EpsEps2) #from dy_GammaGamma
            +np.einsum('ijk,jlkm',self.d2y[Y,S,Z](z_i),d2YprimeGZ_EpsEps2).transpose(0,2,1)
            ,axes=1)
        
        def temp(z_i):
            DFi = DF(z_i)[:-n]
            return -np.linalg.inv(DFi[:,y]+DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy))           
        temp = dict_fun(temp)
        A_i = dict_fun(lambda z_i: np.tensordot(temp(z_i),DFhat_EpsEps(z_i),1))
        B_i = dict_fun(lambda z_i: temp(z_i).dot(DF(z_i)[:-n,Y] + DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat) ))
        C_i = dict_fun(lambda z_i: temp(z_i).dot(DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Y](z_i))))
        A = self.integrate(lambda z_i : np.tensordot(self.dY(z_i).dot(Izy),A_i(z_i),1))
        B,C = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(B_i(z_i))),self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(C_i(z_i)))
        
        tempC = np.linalg.inv(np.eye(nY)-C)
        
        d2y_EE =lambda z_i: A_i(z_i) + np.tensordot(C_i(z_i).dot(tempC),A,1)
        dy_YEE = lambda z_i: B_i(z_i) + C_i(z_i).dot(tempC).dot(B)
        
        HGhat_EE = self.integrate(lambda z_i: HGhat(z_i,Eps,Eps)
        + np.tensordot(self.DG(z_i)[:-nG,y],d2y_EE(z_i),1))
        
        DGhat_YEEinv = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[:-nG,Y] + self.DG(z_i)[:-nG,y].dot(dy_YEE(z_i))))
        
        self.d2Y[Eps,Eps] = -np.tensordot(DGhat_YEEinv,HGhat_EE,1) 
        self.d2y[Eps,Eps] = dict_fun(lambda z_i: d2y_EE(z_i) + np.tensordot(dy_YEE(z_i),self.d2Y[Eps,Eps],1))
        
        #Now derivative with respect to sigma_E
        
        def temp2(z_i):
            DFi,df = DF(z_i)[n:],self.df(z_i)
            return -np.linalg.inv(DFi[:,y]+DFi[:,e].dot(df)+DFi[:,v].dot(Ivy)+DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy))  
        temp2 = dict_fun(temp2)
        
        def A2_i(z_i): #the pure term
            DFi = DF(z_i)[n:]
            df,hf = self.df(z_i),self.Hf(z_i)
            return np.tensordot(temp2(z_i),np.tensordot(DFi[:,e].dot(df),self.d2y[Eps,Eps](z_i),1)
            +np.tensordot(DFi[:,e],quadratic_dot(hf,self.dy[Eps](z_i),self.dy[Eps](z_i)),1)
            ,axes=1)
        A2_i = dict_fun(A2_i)
        B2_i = dict_fun(lambda z_i : temp2(z_i).dot(DF(z_i)[n:,Y] + DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat) ))
        C2_i = dict_fun(lambda z_i : temp2(z_i).dot(DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Y](z_i))) )
        
        
        A2 = self.integrate(lambda z_i : np.tensordot(self.dY(z_i).dot(Izy),A2_i(z_i),1))
        B2,C2 = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(B2_i(z_i))),self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(C2_i(z_i)))
        
        tempC2 = np.linalg.inv(np.eye(nY)-C2)
        
        d2y_sigmaE =lambda z_i: A2_i(z_i) + np.tensordot(C2_i(z_i).dot(tempC2),A2,1)
        dy_YsigmaE = lambda z_i: B2_i(z_i) + C2_i(z_i).dot(tempC2).dot(B2)        
        
        HGhat_sigmaE = self.integrate(lambda z_i: np.tensordot(self.DG(z_i)[nG:,y],d2y_sigmaE(z_i),1))
        
        DGhat_YsigmaEinv = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[nG:,Y] + self.DG(z_i)[nG:,y].dot(dy_YsigmaE(z_i))))
        
        self.d2Y[sigma_E] = - np.tensordot(DGhat_YsigmaEinv,HGhat_sigmaE,1)
        self.d2y[sigma_E] = dict_fun(lambda z_i : d2y_sigmaE(z_i) + np.tensordot(dy_YsigmaE(z_i),self.d2Y[sigma_E],1))
        
        '''Finally derivative with respect to p and Eps'''
        def T(f):
            return self.integrate(lambda z_i: np.tensordot(self.dY(z_i).dot(Izy),f(z_i),1))
        def DF_ytild_pG_inv(z_i):
            DFi = DF(z_i)[:-n]
            return -np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i).dot(Izy)))
        
        c_pG = dict_fun( lambda z_i: DF_ytild_pG_inv(z_i).dot(DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Y](z_i)) ) )
        C_pG = T(c_pG)
        temp_pG = np.linalg.inv(np.eye(nY)-C_pG)
        
        def get_coefficient(DF_Ytild):
            #given a function DF_Ytild compute associated coefficient
            b_i = dict_fun(lambda z_i: np.tensordot(DF_ytild_pG_inv(z_i),DF_Ytild(z_i),1))
            temp = np.tensordot(temp_pG, T(b_i),1)
            return dict_fun(
            lambda z_i: b_i(z_i) +np.tensordot(c_pG(z_i),temp,1)            
            )        
        
        
        def dSprime_p(z_i):
            return np.vstack((Izy.dot(self.dy[p](z_i)),self.dYGamma_p))
        #first integrate derivatives
        z_Eps = dict_fun(lambda z_i: Izy.dot(self.dy[Eps](z_i)))
        z_p = dict_fun(lambda z_i: Izy.dot(self.dy[p](z_i)))
        Yhat_G_Eps = self.integrate(lambda z_i : self.dY(z_i).dot(z_Eps(z_i)))
        Yhat_G_p = self.integrate(lambda z_i: self.dY(z_i).dot(z_p(z_i)))
        Yhat_ZG = self.integrate(lambda z_i : np.tensordot(self.d2Y[Z,z](z_i),z_Eps(z_i),1))
        Yhat_GG1 = self.integrate(lambda z_i :  quadratic_dot(self.d2Y[Y,z](z_i),Yhat_G_p,z_Eps(z_i)))
        Yhat_GG2 = self.integrate(lambda z_i : quadratic_dot(self.d2Y[z,z](z_i),z_p(z_i),z_Eps(z_i)))
        Yhat_pG = self.integrate(lambda z_i : np.tensordot(self.d2Y[p,z](z_i),z_Eps(z_i),1))
        Yhat_YGzpG = self.integrate(lambda z_i : np.einsum('ij,jkl,lm->ikm',self.dY(z_i).dot(Izy),self.d2y[p,z](z_i),z_Eps(z_i)) )
        def DFhat_pEps(z_i):
            dSprime_p_i,dZprime_p = dSprime_p(z_i),IZYhat.dot(self.dY_p)
            dSprime_Eps_i,dZprime_Eps = dSprime_Eps(z_i),IZYhat.dot(self.dY_Eps)
            DFi = DF(z_i)[:-n]
            yprime_ZG = np.einsum('ijk,jkl,km->iml',self.dy[Y,S,Z](z_i),Yhat_ZG,dZprime_p) 
            yprime_GG = np.tensordot(self.dy[Y](z_i),Yhat_GG1 + Yhat_GG2,1)
            yprime_pG = (np.tensordot(self.dytild['pG,Y_pG'](z_i),Yhat_pG,1) 
            + np.einsum('ijkl,jkm->ilm',self.dytild['pG,Y_ZG'](z_i),Yhat_ZG) + np.tensordot(self.dytild['pG,Y_G'](z_i),Yhat_G_Eps,1)
            +np.tensordot(self.dytild['pG,Y_GG'](z_i),Yhat_GG1+Yhat_GG2,1)
            +np.tensordot(self.dytild['pG,Y_Gz_pG'](z_i),Yhat_YGzpG,1))
            return (self.HFhat[p,Eps](z_i)
                +np.tensordot(DFi[:,v].dot(Ivy),quadratic_dot(self.d2y[Z,S](z_i),dZprime_p,dSprime_Eps_i)
                +quadratic_dot(self.d2y[S,Z](z_i),dSprime_p_i,dZprime_Eps)
                + quadratic_dot(self.d2y[S,S](z_i),dSprime_p_i,dSprime_Eps_i)
                +quadratic_dot(self.d2y[Z,Z](z_i),dZprime_p,dZprime_Eps)
                +np.tensordot(self.d2y[p,z](z_i),z_Eps(z_i),1)
                #now for distribution terms
                +yprime_ZG+yprime_GG+ yprime_pG,1) )
        
        DFhat_pEps = dict_fun(DFhat_pEps)
        ytild_pEps = get_coefficient(DFhat_pEps)
        ytild_YpEps = get_coefficient(lambda z_i : DF(z_i)[:-n,Y] + DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat))
        #no solve for Y_pp
        DG_pEps = self.integrate(lambda z_i : HGhat(z_i,p,Eps) + np.tensordot(self.DG(z_i)[:-nG,y],ytild_pEps(z_i),1)  )
        DG_YpEps_inv = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[:-nG,Y] +np.tensordot(self.DG(z_i)[:-nG,y],ytild_YpEps(z_i),1) ))
        self.d2Y[p,Eps] = np.tensordot(DG_YpEps_inv,-DG_pEps,1)
        self.d2Y[Eps,p] = self.d2Y[p,Eps].transpose(0,2,1)
        self.d2y[p,Eps] = dict_fun(lambda z_i: ytild_pEps(z_i) + np.tensordot(ytild_YpEps(z_i),self.d2Y[p,Eps],1))
        self.d2y[Eps,p] = lambda z_i: self.d2y[p,Eps](z_i).transpose(0,2,1)
        
        '''now derivative with respecto Z and Eps'''
        def DFhat_ZEps(z_i):
            dZprime_Z = self.dZ_Z
            dSprime_Eps_i,dZprime_Eps = dSprime_Eps(z_i),IZYhat.dot(self.dY_Eps)
            DFi = DF(z_i)[:-n]
            yprime_ZG = np.einsum('ijk,jkl,km->iml',self.dy[Y,S,Z](z_i),Yhat_ZG,dZprime_Z) 
            return (self.HFhat[Z,Eps](z_i)
                +np.tensordot(DFi[:,v].dot(Ivy),quadratic_dot(self.d2y[Z,S](z_i),dZprime_Z,dSprime_Eps_i)
                +quadratic_dot(self.d2y[Z,Z](z_i),dZprime_Z,dZprime_Eps)
                #now for distribution term
                +yprime_ZG,1) )
                
        DFhat_ZEps = dict_fun(DFhat_ZEps)
        ytild_ZEps = get_coefficient(DFhat_ZEps)
        ytild_YZEps = get_coefficient(lambda z_i : DF(z_i)[:-n,Y] + DF(z_i)[:-n,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat))
        #no solve for Y_pp
        DG_ZEps = self.integrate(lambda z_i : HGhat(z_i,Z,Eps) + np.tensordot(self.DG(z_i)[:-nG,y],ytild_ZEps(z_i),1)  )
        DG_YZEps_inv = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[:-nG,Y] +np.tensordot(self.DG(z_i)[:-nG,y],ytild_YZEps(z_i),1) ))
        self.d2Y[Z,Eps] = np.tensordot(DG_YZEps_inv,-DG_ZEps,1)
        self.d2Y[Eps,Z] = self.d2Y[Z,Eps].transpose(0,2,1)
        self.d2y[Z,Eps] = dict_fun(lambda z_i: ytild_ZEps(z_i) + np.tensordot(ytild_YZEps(z_i),self.d2Y[Z,Eps],1))
        self.d2y[Eps,Z] = lambda z_i: self.d2y[Z,Eps](z_i).transpose(0,2,1)
        
        
    def compute_d2p(self):
        '''
        Coputes all the second order terms w.r.t. parameters
        '''        
        DF = self.DF
        dZGamma_p =IZYhat.dot(self.dYGamma_p)
        def T(f):
            return self.integrate(lambda z_i: np.tensordot(self.dY(z_i).dot(Izy),f(z_i),1))
            
        D = np.diagonal(self.dZ_Z)
        def dSprime_p(z_i):
            return np.vstack((Izy.dot(self.dy[p](z_i)),self.dYGamma_p))
            
        def HGhat(z_i,y1,y2):
            HG = self.HG(z_i)[nG:]
            d = self.get_d(z_i)
            HGhat = np.zeros((nY,len(y1),len(y2)))
            for x1 in [y,z,Y,Z,p]:
                HGx1 = HG[:,x1,:]
                for x2 in [y,z,Y,Z,p]:
                    HGhat += quadratic_dot(HGx1[:,:,x2],d[x1,y1],d[x2,y2])
            return HGhat
        ''' first cross derivative w.r.t. p and Z'''
        d2YprimeGZ_pZ = self.integrate(lambda z_i: np.einsum('ijk,jl,km->ilkm',self.d2Y[z,Z](z_i),Izy.dot(self.dy[p](z_i)),self.dZ_Z))
        
        def DFhat_pZ(z_i):
            dSprime_p_i,dZprime_p,d = dSprime_p(z_i),IZYhat.dot(self.dY_p),self.get_d(z_i)
            DFi,hf = DF(z_i)[n:],self.Hf(z_i)
            return (self.HFhat[p,Z](z_i) + np.tensordot(DFi[:,e],quadratic_dot(hf,d[y,p],d[y,Z]),1)
                +np.tensordot(DFi[:,v].dot(Ivy),quadratic_dot(self.d2y[S,Z](z_i),dSprime_p_i,self.dZ_Z)
                +np.einsum('ijk,jlkm',self.dy[Y,S,Z](z_i),d2YprimeGZ_pZ) #from dy_GammaZ
                +quadratic_dot(self.d2y[Z,Z](z_i),dZprime_p,self.dZ_Z),1 ))
        DFhat_pZ = dict_fun(DFhat_pZ)
            
        def DF_ytild_pZ(z_i):
            DFi,dfi = DF(z_i)[n:],self.df(z_i)
            return (DFi[:,y] + DFi[:,e].dot(dfi) + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i).dot(Izy)))
        DF_ytild_pZ = dict_fun(DF_ytild_pZ)
        DF_Ytild2_pZ = dict_fun(lambda z_i: DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Y](z_i)))
        DF_Y_pZ = dict_fun(lambda z_i: DF(z_i)[n:,Y] + mdot(DF(z_i)[n:,v],Ivy,self.dy[Z](z_i),IZYhat) )
        
        a_pZ = dict_fun(lambda z_i: quadratic_solve3(D,DF(z_i)[n:,v].dot(Ivy),DF_ytild_pZ(z_i),DFhat_pZ(z_i)) ) #ny x n_p x nZ
        b_pZ = dict_fun(lambda z_i: quadratic_solve3(D,DF(z_i)[n:,v].dot(Ivy),DF_ytild_pZ(z_i),DF_Y_pZ(z_i),True)) #ny x nY x nZ
        c_pZ = dict_fun(lambda z_i: quadratic_solve3(D,DF(z_i)[n:,v].dot(Ivy),DF_ytild_pZ(z_i),DF_Ytild2_pZ(z_i),True) )#ny x nY x nZ
        A_pZ = T(a_pZ)#nYxn_pxnZ 
        B_pZ = T(b_pZ)#nYxnYxnZ 
        C_pZ = T(c_pZ)#nYxnYxnZ
        
        temp1_pZ,temp2_pZ = np.empty((nY,n_p,nZ)),np.empty((nY,nY,nZ))
        for i in range(nZ):
            temp = np.linalg.inv(np.eye(nY)-C_pZ[:,:,i])
            temp1_pZ[...,i] = np.tensordot(temp,A_pZ[...,i],1)
            temp2_pZ[:,:,i] = np.tensordot(temp,B_pZ[:,:,i],1)
            
        ytild_pZ = dict_fun(lambda z_i: a_pZ(z_i) + np.einsum('ijk,jlk->ilk',c_pZ(z_i),temp1_pZ))
        ytild_YpZ = dict_fun(lambda z_i: b_pZ(z_i) + np.einsum('ijk,jlk->ilk',c_pZ(z_i),temp2_pZ))
        
        #sovling for YpZ
        DG_pZ = self.integrate(lambda z_i : HGhat(z_i,p,Z) + np.tensordot(self.DG(z_i)[nG:,y],ytild_pZ(z_i),1)  )
        DG_YpZ = self.integrate(lambda z_i : self.DG(z_i)[nG:,Y,np.newaxis] +np.tensordot(self.DG(z_i)[nG:,y],ytild_YpZ(z_i),1) )
        self.d2Y[p,Z] = np.empty((nY,n_p,nZ))
        for i in range(nZ):
            self.d2Y[p,Z][:,:,i] = np.linalg.solve(DG_YpZ[:,:,i],-DG_pZ[:,:,i])
        self.d2y[p,Z] = dict_fun(lambda z_i :ytild_pZ(z_i) + np.einsum('ijk,jlk->ilk',ytild_YpZ(z_i),self.d2Y[p,Z]) )
        
        
        '''Now w.r.t p and Gamma'''
        def DF_ytild_pG_inv(z_i):
            DFi,dfi = DF(z_i)[n:],self.df(z_i)
            return -np.linalg.inv(DFi[:,y] + DFi[:,e].dot(dfi) + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy) + DFi[:,v].dot(Ivy))
        
        c_pG = dict_fun( lambda z_i: DF_ytild_pG_inv(z_i).dot(DF(z_i)[n:,v]).dot(Ivy).dot(self.dy[Y](z_i)) ) 
        C_pG = T(c_pG)
        temp_pG = np.linalg.inv(np.eye(nY)-C_pG)
        
        def get_coefficient(DF_Ytild):
            #given a function DF_Ytild compute associated coefficient
            b_i = dict_fun(lambda z_i: np.tensordot(DF_ytild_pG_inv(z_i),DF_Ytild(z_i),1))
            temp = np.tensordot(temp_pG, T(b_i),1)
            return dict_fun(
            lambda z_i: b_i(z_i) +np.tensordot(c_pG(z_i),temp,1)            
            )
            
        #now compute loadings on the derivatives Y_pG,Y_G, Y_Gz_pG,Y_ZG,Y_GG1,Y_GG2
        self.dytild = {}
        
        self.dytild['pG,Y_pG'] = get_coefficient(
        lambda z_i : DF(z_i)[n:,Y] + DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat)        
        )
        
        self.dytild['pG,Y_ZG'] = get_coefficient(
        lambda z_i : np.tensordot(DF(z_i)[n:,v].dot(Ivy), np.einsum('ijk,kl->ijkl',self.dy[Y,S,Z](z_i),dZGamma_p)
        +np.einsum('ijk,kl->ijkl',self.d2y[Y,S,Z](z_i),IZYhat.dot(self.dY_p)),1)
        )
        
        self.dytild['pG,Y_GG'] = get_coefficient(
        lambda z_i : DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Y](z_i))
        )
        
        #now for Y_G lot of terms
        temp_YGZ = self.integrate(lambda z_i : np.einsum('ijk,jl->ilk',self.d2Y[z,Z](z_i),Izy.dot(self.dy[p](z_i))))
        temp_YGG = self.integrate(lambda z_i : np.einsum('ijk,jl->ilk',self.d2Y[z,Y](z_i),Izy.dot(self.dy[p](z_i))) )
        def DF_ytild_Y_S(z_i):
            DFi,dz_p,hf = self.DF(z_i)[n:],Izy.dot(self.dy[p](z_i)),self.Hf(z_i)
            d = self.get_d(z_i)
            dSprime_p,dZprime_S,dZprime_p = np.vstack((dz_p,self.dYGamma_p)),np.hstack((np.zeros((nZ,nz)),IZYhat)),IZYhat.dot(self.dY_p)
            ret = self.HFhat[p,S](z_i)
            return ret + quadratic_dot(np.tensordot(DFi[:,e],hf,1),d[y,p],d[y,S]) + np.tensordot(DFi[:,v].dot(Ivy),
                                       
            quadratic_dot(self.d2y[S,Z](z_i),dSprime_p,dZprime_S) + np.einsum('ilk,lj->ijk',self.d2y[S,S](z_i),dSprime_p)
            +quadratic_dot(self.d2y[Z,Z](z_i),dZprime_p,dZprime_S) + np.einsum('ilk,lj->ijk',self.d2y[Z,S](z_i),dZprime_p)
            +np.einsum('ijk,jlk,km->ilm',self.dy[Y,S,Z](z_i)+self.d2y[Y,S,Z](z_i),temp_YGZ,dZprime_S)
            +np.einsum('ijk,kl->ijl',self.d2y[p,Z](z_i),dZprime_S)
            +np.tensordot(self.dy[Y](z_i),np.tensordot(temp_YGG,d[Y,S],1),1)
            ,1)
            
        DF_ytild_Y_S = dict_fun(DF_ytild_Y_S)
        self.dytild['pG,Y_G'] = get_coefficient(lambda z_i: DF_ytild_Y_S(z_i)[:,:,nz:])
        self.d2y[p,z] = dict_fun(lambda z_i: np.tensordot(DF_ytild_pG_inv(z_i),DF_ytild_Y_S(z_i)[:,:,:nz],1))
        self.d2y[z,p] = lambda z_i : self.d2y[p,z].transpoze(0,2,1)


        self.dytild['pG,Y_Gz_pG'] = get_coefficient(
        lambda z_i : DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Y](z_i))
        )
        
        
        #now compute Y_pG
        DGi = lambda z_i : self.DG(z_i)[nG:]
        HGhat_pS = dict_fun(lambda z_i : HGhat(z_i,p,S))
        HGhat_pz = lambda z_i : HGhat_pS(z_i)[:,:,:nz] + np.tensordot(DGi(z_i)[:,y],self.d2y[p,z](z_i),1)
        DG_pG_YpG = self.integrate(lambda z_i : np.tensordot(DGi(z_i)[:,y],self.dytild['pG,Y_pG'](z_i),1) + DGi(z_i)[:,Y])
        DG_pG_YZG = self.integrate(lambda z_i : np.tensordot(DGi(z_i)[:,y],self.dytild['pG,Y_ZG'](z_i),1))
        DG_pG_YGG = self.integrate(lambda z_i : np.tensordot(DGi(z_i)[:,y],self.dytild['pG,Y_GG'](z_i),1))
        DG_pG_YG = self.integrate(lambda z_i : np.tensordot(DGi(z_i)[:,y],self.dytild['pG,Y_G'](z_i),1) + HGhat_pS(z_i)[:,:,nz:]) 
        DG_pG_YGzpG = self.integrate(lambda z_i : np.tensordot(DGi(z_i)[:,y],self.dytild['pG,Y_Gz_pG'](z_i),1))
        
        
        def HG_pz(z_i):
            return (HGhat_pz(z_i) + np.einsum('ijkl,jkm->ilm',DG_pG_YZG,self.d2Y[Z,z](z_i))
            +np.tensordot(DG_pG_YGG,quadratic_dot(self.d2Y[z,z](z_i),Izy.dot(self.dy[p](z_i)),np.eye(nz)) + quadratic_dot(self.d2Y[Y,z](z_i),self.dYGamma_p,np.eye(nz)),1)
            +np.tensordot(DG_pG_YG,self.dY(z_i),1) + np.tensordot(DG_pG_YGzpG, np.tensordot(self.dY(z_i).dot(Izy),self.d2y[p,z](z_i),1) ,1)            
            )
        self.d2Y[p,z] = dict_fun(lambda z_i : np.tensordot(np.linalg.inv(DG_pG_YpG), -HG_pz(z_i),1) ) 
        self.d2Y[z,p] = lambda z_i : self.d2Y[z,p](z_i).transpose(0,2,1)
        
        '''Finally derivative with respect to p and p'''
        #first integrate derivatives
        z_p = dict_fun(lambda z_i: Izy.dot(self.dy[p](z_i)))
        Yhat_G = self.integrate(lambda z_i : self.dY(z_i).dot(z_p(z_i)))
        Yhat_ZG = self.integrate(lambda z_i : np.tensordot(self.d2Y[Z,z](z_i),z_p(z_i),1))
        Yhat_GG1 = self.integrate(lambda z_i :  quadratic_dot(self.d2Y[Y,z](z_i),Yhat_G,z_p(z_i)))
        Yhat_GG2 = self.integrate(lambda z_i : quadratic_dot(self.d2Y[z,z](z_i),z_p(z_i),z_p(z_i)))
        Yhat_pG = self.integrate(lambda z_i : np.tensordot(self.d2Y[p,z](z_i),z_p(z_i),1))
        Yhat_YGzpG = self.integrate(lambda z_i : np.einsum('ij,jkl,lm->ikm',self.dY(z_i).dot(Izy),self.d2y[p,z](z_i),z_p(z_i)) )
        def DFhat_pp(z_i):
            dSprime_p_i,dZprime_p,d = dSprime_p(z_i),IZYhat.dot(self.dY_p),self.get_d(z_i)
            DFi,hf = DF(z_i)[n:],self.Hf(z_i)
            yprime_ZG = np.einsum('ijk,jkl,km->iml',self.dy[Y,S,Z](z_i),Yhat_ZG,dZprime_p) 
            yprime_GG = np.tensordot(self.dy[Y](z_i),Yhat_GG1+ Yhat_GG1.transpose(0,2,1) + Yhat_GG2,1)
            yprime_pG = (np.tensordot(self.dytild['pG,Y_pG'](z_i),Yhat_pG,1) 
            + np.einsum('ijkl,jkm->ilm',self.dytild['pG,Y_ZG'](z_i),Yhat_ZG) + np.tensordot(self.dytild['pG,Y_G'](z_i),Yhat_G,1)
            +np.tensordot(self.dytild['pG,Y_GG'](z_i),Yhat_GG1+Yhat_GG2,1)
            +np.tensordot(self.dytild['pG,Y_Gz_pG'](z_i),Yhat_YGzpG,1))
            return (self.HFhat[p,p](z_i) + np.tensordot(DFi[:,e],quadratic_dot(hf,d[y,p],d[y,p]),1)
                +np.tensordot(DFi[:,v].dot(Ivy),transpose_sum(quadratic_dot(self.d2y[Z,S](z_i),dZprime_p,dSprime_p_i))
                + quadratic_dot(self.d2y[S,S](z_i),dSprime_p_i,dSprime_p_i)
                +quadratic_dot(self.d2y[Z,Z](z_i),dZprime_p,dZprime_p)
                +transpose_sum(np.tensordot(self.d2y[p,z](z_i),Izy.dot(self.dy[p](z_i)),1))
                #now for distribution terms
                +transpose_sum(yprime_ZG)+yprime_GG+ transpose_sum(yprime_pG),1) )
        DFhat_pp = dict_fun(DFhat_pp)
        ytild_pp = get_coefficient(DFhat_pp)
        ytild_Ypp = get_coefficient(lambda z_i : DF(z_i)[n:,Y] + DF(z_i)[n:,v].dot(Ivy).dot(self.dy[Z](z_i)).dot(IZYhat))
        #no solve for Y_pp
        DG_pp = self.integrate(lambda z_i : HGhat(z_i,p,p) + np.tensordot(self.DG(z_i)[nG:,y],ytild_pp(z_i),1)  )
        DG_Ypp_inv = np.linalg.inv(self.integrate(lambda z_i : self.DG(z_i)[nG:,Y] +np.tensordot(self.DG(z_i)[nG:,y],ytild_Ypp(z_i),1) ))
        self.d2Y[p,p] = np.tensordot(DG_Ypp_inv,-DG_pp,1)
        self.d2y[p,p] = dict_fun(lambda z_i: ytild_pp(z_i) + np.tensordot(ytild_Ypp(z_i),self.d2Y[p,p],1))
        
        
    def test(self):
        z_i = self.Gamma[0]# assume only 1
        DFi,df,hf = self.DF(z_i)[n:],self.df(z_i),self.Hf(z_i)
        ypG = (np.tensordot(self.dytild['pG,Y_pG'],self.d2Y[p,z](z_i),1) + np.einsum('ijkl,jkm->ilm',self.dytild['pG,Y_ZG'](z_i),self.d2Y[Z,z](z_i))
        +np.tensordot(self.dytild['pG,Y_G'],self.dY(z_i),1) + np.tensordot(self.ytild['pG,Y_GG'],np.einsum('ijk,jl->ilk',self.d2Y[Y,z](z_i),self.dY_Gamma_p)+ np.einsum('ijk,jl->ilk',self.d2Y[z,z](z_i),Izy.dot(self.dy[p](z_i))),1)
        +np.tensordot(self.dytild['pG,Y_Gz_pG'],np.tensordot(self.dY(z_i).dot(Izy),self.d2y[p,z](z_i),1),1)
        )
        
        
    def compute_d2y(self):
        '''
        Computes second derivative of y
        '''
        #DF,HF = self.DF(z_i),self.HF(z_i)
        #df,Hf = self.df(z_i),self.Hf(z_i)
        
        #first compute DFhat, need loadings of S and epsilon on variables
                
        #Now compute d2y
        
        def d2y_SS(z_i):
            DF,df,Hf = self.DF(z_i),self.df(z_i),self.Hf(z_i)
            d = self.get_d(z_i)
            DFi = DF[n:]
            temp = np.einsum('ij...,j...->i...',DFi[:,v].dot(Ivy),quadratic_dot(self.d2y[Z,S](z_i),IZYhat.dot(d[Y,S]),np.eye(nY+nz)))
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy)),
                -self.HFhat[S,S](z_i) - np.tensordot(DFi[:,e],quadratic_dot(Hf,d[y,S],d[y,S]),1)
                -np.tensordot(DFi[:,v].dot(Ivy),quadratic_dot(self.d2y[Z,Z](z_i),IZYhat.dot(d[Y,S]),IZYhat.dot(d[Y,S])),1)
                -temp - temp.transpose(0,2,1)
                , axes=1)
        self.d2y[S,S] = dict_fun(d2y_SS)
        
        def d2y_YSZ(z_i):
            DFi,df = self.DF(z_i)[n:],self.df(z_i)
            temp = np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy))
            return - np.einsum('ij...,j...->i...',temp.dot(DFi[:,v]).dot(Ivy), self.dy[Y,S,Z](z_i))
        self.d2y[Y,S,Z] = dict_fun(d2y_YSZ)
        
        def d2y_Seps(z_i):
            DF = self.DF(z_i)
            d = self.get_d(z_i)
            DFi = DF[:-n]
            dz_eps= Izy.dot(d[y,eps])
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
            -self.HFhat[S,eps](z_i) - np.tensordot(DFi[:,v].dot(Ivy), self.d2y[S,S](z_i)[:,:,:nz].dot(dz_eps),1)
            -np.tensordot(DFi[:,v].dot(Ivy), quadratic_dot(self.d2y[Z,S](z_i)[:,:,:nz],IZYhat.dot(d[Y,S]),dz_eps),1)
            , axes=1)
            
        self.d2y[S,eps] = dict_fun(d2y_Seps)
        self.d2y[eps,S] = dict_fun(lambda z_i : self.d2y[S,eps](z_i).transpose(0,2,1))
        
        def d2y_epseps(z_i):
            DF = self.DF(z_i)
            d = self.get_d(z_i)
            DFi = DF[:-n]
            dz_eps= Izy.dot(d[y,eps])
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
            -self.HFhat[eps,eps](z_i) - np.tensordot(DFi[:,v].dot(Ivy), 
            quadratic_dot(self.d2y[S,S](z_i)[:,:nz,:nz],dz_eps,dz_eps),1)
            ,axes=1)
            
        self.d2y[eps,eps] = dict_fun(d2y_epseps)
        
    
        
        
    def quadratic(self):
        '''
        Computes the quadratic approximation
        '''
        self.d2Y,self.d2y = {},{}
        self.dYGamma_Eps = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(self.dy[Eps](z_i)))
        self.dYGamma_p = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(self.dy[p](z_i)))
        self.get_d = dict_fun(self.get_df)
        self.compute_HFhat()
        self.compute_d2y_ZZ()
        self.compute_d2y()        
        #Now d2Y
        DGhat_f = dict_fun(self.compute_DGhat)
        Temp = self.integrate( lambda z_i: np.einsum('ij...,j...->i...',self.DG(z_i)[nG:,y],self.d2y[Y,S,Z](z_i)) )
        def DGhat_zY(z_i):
            temp2 = np.einsum('ijk,jlk->ilk',Temp,self.d2Y[z,Z](z_i)) #don't sum over Z axis
            return  DGhat_f(z_i)[:,:nz,nz:] + np.einsum('ilk,km',temp2,IZYhat)
            
        self.DGhat = {}
        self.DGhat[z,z] = lambda z_i : DGhat_f(z_i)[:,:nz,:nz]
        self.DGhat[z,Y] = dict_fun(DGhat_zY)
        self.DGhat[Y,z] = lambda z_i : self.DGhat[z,Y](z_i).transpose(0,2,1)
        self.DGhat[Y,Y] = self.integrate(lambda z_i : DGhat_f(z_i)[:,nz:,nz:])
        self.compute_d2Y()
        self.compute_dsigma()
        self.compute_d2p()
        self.compute_d2y_Eps()

                
    def compute_d2Y(self):
        '''
        Computes components of d2Y
        '''
        DGhat = self.DGhat
        #First z_i,z_i
        self.d2Y[z,z] = dict_fun(lambda z_i: np.tensordot(self.DGYinv, - DGhat[z,z](z_i),1))
            
        self.d2Y[Y,z] = dict_fun(lambda z_i: np.tensordot(self.DGYinv, - DGhat[Y,z](z_i) 
                            -DGhat[Y,Y].dot(self.dY(z_i))/2. ,1) )
            
        self.d2Y[z,Y] = lambda z_i : self.d2Y[Y,z](z_i).transpose(0,2,1)
        
            
    def compute_DGhat(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i)[nG:,:],self.HG(z_i)[nG:,:,:]
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros((nY,nz+nY,nz+nY))
        DGhat += np.tensordot(DG[:,y],d2y[S,S](z_i),1)
        for x1 in [y,z,Y,Z]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,z,Y,Z]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,S],d[x2,S])
        return DGhat
        
    def compute_d2y_sigma(self,z_i):
        '''
        Computes linear contribution of sigma, dYsigma and dY1sigma
        '''
        DF = self.DF(z_i)
        df,Hf = self.df(z_i),self.Hf(z_i)
        #first compute DFhat, need loadings of S and epsilon on variables
        d = self.get_d(z_i)
        d[y,Y] = d[y,S][:,nz:]
        DFi = DF[n:] #conditions like x_i = Ex_i don't effect this
        
        
        temp = np.linalg.inv(DFi[:,y]+DFi[:,e].dot(df)+DFi[:,v].dot(Ivy+Ivy.dot(d[y,z]).dot(Izy)))
        
        Ahat = (-DFi[:,e].dot(df).dot(self.d2y[eps,eps](z_i).diagonal(0,1,2)) 
        -DFi[:,e].dot(quadratic_dot(Hf,d[y,eps],d[y,eps]).diagonal(0,1,2)) 
        -DFi[:,v].dot(Ivy).dot(d[y,Y]).dot(self.integral_term) )
        
        Bhat = -DFi[:,Y]- DFi[:,v].dot(Ivy).dot(d[y,Z]).dot(IZYhat)
        Chat = -DFi[:,v].dot(Ivy).dot(d[y,Y])
        return temp.dot(np.hstack((Ahat.reshape(-1,neps),Bhat,Chat)))
        
    def compute_DGhat_sigma(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i)[nG:,:],self.HG(z_i)[nG:,:,:]
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros((nY,neps))
        DGhat += DG[:,y].dot(d2y[eps,eps](z_i).diagonal(0,1,2))
        d[eps,eps] =np.eye(neps)
        for x1 in [y,eps]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,eps]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,eps],d[x2,eps]).diagonal(0,1,2)
        return DGhat
        
    def compute_dsigma(self):
        '''
        Computes how dY and dy_i depend on sigma
        '''
        DG = lambda z_i : self.DG(z_i)[nG:,:]
        #Now how do things depend with sigma
        self.integral_term =self.integrate(lambda z_i:
            quadratic_dot(self.d2Y[z,z](z_i),Izy.dot(self.dy[eps](z_i)),Izy.dot(self.dy[eps](z_i))).diagonal(0,1,2)
            + self.dY(z_i).dot(Izy).dot(self.d2y[eps,eps](z_i).diagonal(0,1,2)))
            
        ABCi = dict_fun(self.compute_d2y_sigma )
        Ai,Bi,Ci = lambda z_i : ABCi(z_i)[:,:neps], lambda z_i : ABCi(z_i)[:,neps:nY+neps], lambda z_i : ABCi(z_i)[:,nY+neps:]
        Atild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ai(z_i)))
        Btild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Bi(z_i)))
        Ctild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ci(z_i)))
        tempC = np.linalg.inv(np.eye(nY)-Ctild)

        DGhat = self.integrate(self.compute_DGhat_sigma)
        
        temp1 = self.integrate(lambda z_i:DG(z_i)[:,Y] + DG(z_i)[:,y].dot(Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild)) )
        temp2 = self.integrate(lambda z_i:DG(z_i)[:,y].dot(Ai(z_i)+Ci(z_i).dot(tempC).dot(Atild)) )
        
        self.d2Y[sigma] = np.linalg.solve(temp1,-DGhat-temp2)
        self.d2y[sigma] = dict_fun(lambda z_i: Ai(z_i) + Ci(z_i).dot(tempC).dot(Atild) +
                      ( Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild) ).dot(self.d2Y[sigma]))
                      
                      
    def join_function(self):
        '''
        Joins the data for the dict_maps across functions
        '''
        fast = len(self.dist)%size == 0
        for f in self.dy.values():
            if hasattr(f, 'join'):
                parallel_map_noret(lambda x: f(x[0]),self.dist)
                f.join(fast)
        for f in self.d2y.values():
            if hasattr(f,'join'):       
                parallel_map_noret(lambda x: f(x[0]),self.dist)
                f.join(fast)
          
        self.dY.join(fast)
        
        for f in self.d2Y.values():
            if hasattr(f,'join'):   
                parallel_map_noret(lambda x: f(x[0]),self.dist)
                f.join(fast)
        
        
    def iterate(self,Zbar,quadratic = True):
        '''
        Iterates the distribution by randomly sampling
        '''
        if not self.simple:
            return self.iterate_complex(Zbar,quadratic)
        else:
            return self.iterate_simple(Zbar)
            
            
    def iterate_complex(self,Zbar,quadratic):
        Zhat = self.dZhat_Z.dot(Zbar-self.ss.get_Y()[:nZ])
        if rank == 0:
            r = np.random.randn()
            if not shock == None:
                r = shock
            r = min(3.,max(-3.,r))
            E = r*sigma_E
        else:
            E = None
            
        E = comm.bcast(E)
        phat = Para.phat
        Gamma_dist = zip(self.Gamma,self.Gamma_ss)
        
        Y1hat = parallel_sum(lambda z : self.dY(z[1]).dot((z[0]-z[1]))
                          ,Gamma_dist)/len(self.Gamma)
        def Y2_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return (quadratic_dot(self.d2Y[z,z](zbar),zhat,zhat) + 2* quadratic_dot(self.d2Y[z,Y](zbar),zhat,Y1hat)).flatten()
        
        Y2hat = parallel_sum(Y2_int,Gamma_dist)/len(self.Gamma)
        
        def Y2_GZ_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return quadratic_dot(self.d2Y[z,Z](zbar),zhat,np.eye(nZ)).reshape(nY,nZ)
        Y2hat_GZ =  parallel_sum(Y2_GZ_int,Gamma_dist)/len(self.Gamma)
        
        def compute_ye(x):
            z_i,zbar = x
            extreme = Para.check_extreme(z_i)
            zhat = z_i-zbar
            r = np.random.randn(neps)
            for i in range(neps):
                r[i] = min(3.,max(-3.,r[i]))
            e = r*sigma
            Shat = np.hstack([zhat,Y1hat])
            if not extreme:
                return np.hstack(( self.ss.get_y(zbar).flatten() + self.dy[eps](zbar).dot(e).flatten() + (self.dy[Eps](zbar).flatten())*E
                                    + self.dy[p](zbar).dot(phat).flatten()
                                    + self.dy[z](zbar).dot(zhat).flatten()
                                    + self.dy[Y](zbar).dot(Y1hat).flatten()
                                    + self.dy[Z](zbar).dot(Zhat).flatten()
                                    + 0.5*quadratic*(quadratic_dot(self.d2y[eps,eps](zbar),e,e).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                    +quadratic_dot(self.d2y[S,S](zbar),Shat,Shat) + 2*quadratic_dot(self.d2y[S,eps](zbar),Shat,e)
                                    +self.dy[Y](zbar).dot(Y2hat)      
                                    +2*quadratic_dot(self.d2y[Z,S](zbar),Zhat,Shat)
                                    +2*quadratic_dot(self.d2y[Z,eps](zbar),Zhat,e)
                                    +quadratic_dot(self.d2y[Z,Z](zbar),Zhat,Zhat)
                                    +2*np.einsum('ijk,jk,k',self.dy[Y,S,Z](zbar),Y2hat_GZ,Zhat)
                                    +2*np.einsum('ijk,jk,k',self.d2y[Y,S,Z](zbar),Y2hat_GZ,IZYhat.dot(Y1hat))
                                    +self.d2y[Eps,Eps](zbar).flatten()*E**2
                                    +self.d2y[sigma_E](zbar).flatten()*sigma_E**2
                                    ).flatten()
                                   ,e))
            else:
                return np.hstack(( self.ss.get_y(zbar).flatten()  + self.dy[Eps](zbar).flatten()*E
                                    + self.dy[p](zbar).dot(phat).flatten()
                                    + self.dy[z](zbar).dot(zhat).flatten()
                                    + self.dy[Y](zbar).dot(Y1hat).flatten()
                                    + self.dy[Z](zbar).dot(Zhat).flatten()
                                    + 0.5*quadratic*(quadratic_dot(self.d2y[eps,eps](zbar),sigma,sigma).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                    +quadratic_dot(self.d2y[S,S](zbar),Shat,Shat)
                                    +self.dy[Y](zbar).dot(Y2hat)      
                                    +2*quadratic_dot(self.d2y[Z,S](zbar),Zhat,Shat)
                                    +quadratic_dot(self.d2y[Z,Z](zbar),Zhat,Zhat)
                                    +2*np.einsum('ijk,jk,k',self.dy[Y,S,Z](zbar),Y2hat_GZ,Zhat)
                                    +2*np.einsum('ijk,jk,k',self.d2y[Y,S,Z](zbar),Y2hat_GZ,IZYhat.dot(Y1hat))
                                    +self.d2y[Eps,Eps](zbar).flatten()*E**2
                                    +self.d2y[sigma_E](zbar).flatten()*sigma_E**2
                                    ).flatten()
                                   ,0.*e))
        if rank == 0:    
            ye = np.vstack(parallel_map(compute_ye,Gamma_dist))
            y,epsilon = ye[:,:-neps],ye[:,-neps]
            Gamma = y.dot(Izy.T)
            Ynew = (self.ss.get_Y() + Y1hat + self.dY_Eps.flatten()*E
                    + self.dY_p.dot(phat).flatten()
                    + self.dY_Z.dot(Zhat)
                    + 0.5*quadratic*(self.d2Y[sigma].dot(sigma**2) + Y2hat + 2*Y2hat_GZ.dot(Zhat)).flatten()
                    + 0.5*(self.d2Y[Eps,Eps].flatten()*E**2 + self.d2Y[sigma_E].flatten()*sigma_E**2)
                    + 0.5*quadratic_dot(self.d2Y[Z,Z],Zhat,Zhat))
                    
            Znew = Ynew[:nZ]
            return Gamma,Znew,Ynew,epsilon,y
        else:
            parallel_map(compute_ye,Gamma_dist)
            return None
            
    def iterate_simple(self,Zbar):
        Zhat = self.dZhat_Z.dot(Zbar-self.ss.get_Y()[:nZ])
        cov_E = sigma_E.dot(sigma_E.T)
        if rank == 0:
            r = np.random.randn()
            if not shock == None:
                r = shock
            r = min(3.,max(-3.,r))
            E = r*sigma_E
            E = np.random.multivariate_normal(np.zeros(nEps),cov_E)
        else:
            E = None
            
        E = comm.bcast(E)
        phat = Para.phat
        
        def compute_ye(x):
            zbar = x
            r = np.random.randn(neps)
            for i in range(neps):
                r[i] = min(3.,max(-3.,r[i]))
            e = r*sigma
            return np.hstack(( self.ss.get_y(zbar).flatten() + self.dy[eps](zbar).dot(e).flatten() + self.dy[Eps](zbar).dot(E).flatten()
                                + self.dy[p](zbar).dot(phat).flatten()
                                + self.dy[Z](zbar).dot(Zhat).flatten()
                                + 0.5*(quadratic_dot(self.d2y[eps,eps](zbar),e,e).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                +2*quadratic_dot(self.d2y[Z,eps](zbar),Zhat,e).flatten()
                                +2*quadratic_dot(self.d2y[Z,Eps](zbar),Zhat,E).flatten()
                                +quadratic_dot(self.d2y[Z,Z](zbar),Zhat,Zhat).flatten()
                                +2*quadratic_dot(self.d2y[p,Z](zbar),phat,Zhat).flatten()
                                +2*quadratic_dot(self.d2y[p,Eps](zbar),phat,E).flatten()
                                +quadratic_dot(self.d2y[p,p](zbar),phat,phat).flatten()
                                +quadratic_dot(self.d2y[Eps,Eps](zbar),E,E).flatten()
                                +np.einsum('ijk,jk',self.d2y[sigma_E](zbar),cov_E).flatten()
                                ).flatten()
                               ,e))
                               
        if rank == 0:    
            ye = np.vstack(parallel_map(compute_ye,self.Gamma))
            y,epsilon = ye[:,:-neps],ye[:,-neps:]
            Gamma = y.dot(Izy.T)
            Ynew = (self.ss.get_Y().flatten()+ self.dY_Eps.dot(E).flatten()
                    + self.dY_p.dot(phat).flatten()
                    + self.dY_Z.dot(Zhat).flatten()
                    + 0.5*(self.d2Y[sigma].dot(sigma**2).flatten()
                    + quadratic_dot(self.d2Y[Eps,Eps],E,E).flatten()
                    + np.einsum('ijk,jk',self.d2Y[sigma_E],cov_E).flatten()
                    + quadratic_dot(self.d2Y[Z,Z],Zhat,Zhat).flatten()
                    + 2*quadratic_dot(self.d2Y[Z,Eps],Zhat,E).flatten()
                    + 2*quadratic_dot(self.d2Y[p,Z],phat,Zhat).flatten()
                    + 2*quadratic_dot(self.d2Y[p,Eps],phat,E).flatten()
                    + quadratic_dot(self.d2Y[p,p],phat,phat).flatten()))
                    
            Znew = Ynew[:nZ]
            return Gamma,Znew,Ynew,epsilon,y
        else:
            parallel_map(compute_ye,self.Gamma)
            return None

    def iterate_ConditionalMean(self,Zbar,quadratic = True):
        '''
        Iterates the distribution by randomly sampling
        '''
        Zhat = self.dZhat_Z.dot(Zbar-self.ss.get_Y()[:nZ])
        if rank == 0:
            r = np.random.randn()
            if not shock == None:
                r = shock
            r = min(3.,max(-3.,r))
            E = r*sigma_E
        else:
            E = None
            
        E = comm.bcast(E)
        phat = Para.phat
        Gamma_dist = zip(self.Gamma,self.Gamma_ss)
        
        Y1hat = parallel_sum(lambda z : self.dY(z[1]).dot((z[0]-z[1]))
                          ,Gamma_dist)/len(self.Gamma)
        def Y2_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return (quadratic_dot(self.d2Y[z,z](zbar),zhat,zhat) + 2* quadratic_dot(self.d2Y[z,Y](zbar),zhat,Y1hat)).flatten()
        
        Y2hat = parallel_sum(Y2_int,Gamma_dist)/len(self.Gamma)
        
        def Y2_GZ_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return quadratic_dot(self.d2Y[z,Z](zbar),zhat,np.eye(nZ)).reshape(nY,nZ)
        Y2hat_GZ =  parallel_sum(Y2_GZ_int,Gamma_dist)/len(self.Gamma)
        
        def compute_ye(x):
            z_i,zbar = x
            extreme = Para.check_extreme(z_i)
            zhat = z_i-zbar
            r = np.random.randn(neps)
            for i in range(neps):
                r[i] = min(3.,max(-3.,r[i]))
            e = r*sigma
            Shat = np.hstack([zhat,Y1hat])
            if not extreme:
                return np.hstack(( self.ss.get_y(zbar).flatten() + self.dy[eps](zbar).dot(e).flatten() + (self.dy[Eps](zbar).flatten())*0.
                                    + self.dy[p](zbar).dot(phat).flatten()
                                    + self.dy[z](zbar).dot(zhat).flatten()
                                    + self.dy[Y](zbar).dot(Y1hat).flatten()
                                    + self.dy[Z](zbar).dot(Zhat).flatten()
                                    + 0.5*quadratic*(quadratic_dot(self.d2y[eps,eps](zbar),e,e).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                    +quadratic_dot(self.d2y[S,S](zbar),Shat,Shat) + 2*quadratic_dot(self.d2y[S,eps](zbar),Shat,e)
                                    +self.dy[Y](zbar).dot(Y2hat)      
                                    +2*quadratic_dot(self.d2y[Z,S](zbar),Zhat,Shat)
                                    +2*quadratic_dot(self.d2y[Z,eps](zbar),Zhat,e)
                                    +quadratic_dot(self.d2y[Z,Z](zbar),Zhat,Zhat)
                                    +2*np.einsum('ijk,jk,k',self.dy[Y,S,Z](zbar),Y2hat_GZ,Zhat)
                                    +2*np.einsum('ijk,jk,k',self.d2y[Y,S,Z](zbar),Y2hat_GZ,IZYhat.dot(Y1hat))
                                    +self.d2y[Eps,Eps](zbar).flatten()*sigma_E**2
                                    +self.d2y[sigma_E](zbar).flatten()*sigma_E**2
                                    ).flatten()
                                   ,e))
        if rank == 0:    
            ye = np.vstack(parallel_map(compute_ye,Gamma_dist))
            y,epsilon = ye[:,:-neps],ye[:,-neps]
            Gamma = y.dot(Izy.T)
            Ynew = (self.ss.get_Y() + Y1hat + self.dY_Eps.flatten()*0.
                    + self.dY_p.dot(phat).flatten()
                    + self.dY_Z.dot(Zhat)
                    + 0.5*quadratic*(self.d2Y[sigma].dot(sigma**2) + Y2hat + 2*Y2hat_GZ.dot(Zhat)).flatten()
                    + 0.5*(self.d2Y[Eps,Eps].flatten()*sigma_E**2 + self.d2Y[sigma_E].flatten()*sigma_E**2)
                    + 0.5*quadratic_dot(self.d2Y[Z,Z],Zhat,Zhat))
            Znew = Ynew[:nZ]
            return Gamma,Znew,Ynew,epsilon,y
        else:
            parallel_map(compute_ye,Gamma_dist)
            return None
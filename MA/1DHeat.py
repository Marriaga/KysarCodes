import scipy.sparse as sps
import numpy as np


def GetGaussPoints(ngaus):
    weights = np.zeros(ngaus)
    positions = np.zeros(ngaus)

    if ngaus==1:
        weights[0] = 2.0
        positions[0] = 0.0
    elif ngaus==2:
        weights[0] =  1.0
        weights[1] =  1.0
        positions[0] = -0.577350269189626
        positions[1] =  0.577350269189626
    elif ngaus==3:
        weights[0] =  0.5555555555555556
        weights[1] =  0.8888888888888889
        weights[2] =  0.5555555555555556
        positions[0] = -0.774596669241483
        positions[1] =  0.0
        positions[2] =  0.774596669241483
    elif ngaus==4:
        weights[0] =  0.347854845137454
        weights[1] =  0.652145154862546
        weights[2] =  0.652145154862546
        weights[3] =  0.347854845137454
        positions[0] = -0.861136311594053
        positions[1] = -0.339981043584856 
        positions[2] =  0.339981043584856
        positions[3] =  0.861136311594053

    return weights,positions


class SF_Linear1D(object):
    def __init__(self,h):
        self.h = h
        self.J = 1/h
        self.Nnodes

    def N(self,eta):
        return np.array([1-eta,1+eta],ndmin=2).T/2

    def B(self,eta):
        return 2/self.h*np.array([-1,1],ndmin=2).T/2

class GlobalParameters(object):
    def __init__(self):
        self.CurrTime = 0
        self.dt = 0.1
        self.Ngp = 2
        self.TimeIntType = "FirstOrder" #"Newmark"
        self.TimeIntParameters = (1.0,0.0) #(Beta,Gamma)=(0.25,0.5)
        self.MCKfacts = self.Get_MCKfacts()

    def Get_MCKfacts(self):
        if self.TimeIntType == "Newmark":
            beta,gamma = self.TimeIntParameters
            MCKfacts = ( 1/(beta*self.dt*self.dt) , gamma/(beta*self.dt) , 1.0 )
        elif self.TimeIntType == "FirstOrder":
            alpha,_ = self.TimeIntParameters
            MCKfacts = ( 0.0 , alpha/self.dt , 1.0 )

    def UpdateTime(self):
        self.CurrTime += self.dt


class LocalParameters(object):
    def __init__(self):
        self.DifK = 2.0
        self.Rho = 2.0

class Connectivity(object):
    def __init__(self,N):
        self.N = N
        self.ConnectivityMatrix = None
        self.ComputeConnectivityMatrix()
    def ComputeConnectivityMatrix(self):
        self.ConnectivityMatrix = np.zeros(self.N-1,dtype=[('ni','<i4'),('nf','<i4')])
        self.ConnectivityMatrix['ni'] = np.arange(0,self.N-1)
        self.ConnectivityMatrix['nf'] = np.arange(1,self.N)
    def Nodes(self,e):
        return list(self.ConnectivityMatrix[e])

def Element_Heat1D(Xe,Ue,GlobalP,LocalP):
    # Gauss Points
    Ngp = GlobalP.Ngp
    gp_w,gp_p = GetGaussPoints(Ngp)

    # Time Factors
    Mfact,Cfact,Kfact = GlobalP.MCKfacts

    # 
    h = abs(Xe[1]-Xe[0])
    shp = SF_Linear1D(h)
    J = shp.J
    Nnodes = shp.Nnodes

    KT = np.zeros((Nnodes,Nnodes))
    #F = np.zeros((Nnodes,1))

    for gp in range(Ngp):
        w = gp_w[gp]
        eta = gp_p[gp]
        N = shp.N(eta)
        B = shp.B(eta)

        KT += w*(Cfact*N.T*LocalP.Rho*N + Kfact*B.T*LocalP.DifK*B)*J
        #F += w*(N.T*force)*J
        
    return KT

def SparseIndex(e,i,j,ElmMatEq):
    return e*ElmMatEq*ElmMatEq + i*ElmMatEq + j



L=5    # m
N=101  
Nelm = N-1

GlobalP = GlobalParameters()
LocalP = LocalParameters()

X=np.linspace(0,L,N)
U=np.zeros(N)

Conn = Connectivity(N)

ElmMatEq = 2
NElmVals = ElmMatEq*ElmMatEq


row = np.zeros(Nelm*NElmVals)
col = np.zeros(Nelm*NElmVals)
for e in range(Nelm):
    Nodes = Conn.Nodes(e)
    for i in range(ElmMatEq):
        for j in range(ElmMatEq):
            row[SparseIndex(e,i,j,ElmMatEq)] = Nodes[i]
            col[SparseIndex(e,i,j,ElmMatEq)] = Nodes[j]


# Make Global Jacobian
val = np.zeros(Nelm*NElmVals)
for e in range(Nelm):
    Nodes = Conn.Nodes(e)
    Ke = Element_Heat1D(X[Nodes],U[Nodes],GlobalP,LocalP)
    for i in range(ElmMatEq):
        for j in range(ElmMatEq):
            val[SparseIndex(e,i,j,ElmMatEq)] = Ke[i,j]
JACOBIAN = sps.coo_matrix((val, (row, col)), shape=(N,N)).tolil()

# Adjust BC
JACOBIAN[0,:] = 0
JACOBIAN[0,0] = 1
JACOBIAN[-1,:] = 0
JACOBIAN[-1,-1] = 1
JACOBIAN = JACOBIAN.tocsr()










row = np.array()


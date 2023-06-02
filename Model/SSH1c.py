# This is SSH non-Local model
# see https://doi.org/10.1063/1.3604561
import numpy as np
from scipy.sparse import csr_matrix

class parameters():
   amu, ps, cm, Å =  1836.0, 41341.37, 1/0.000004556, 1.8897259885789
   c, π = 137.0, np.pi
   dtN = 100
   NSteps = 4200*3//dtN #int(2*10**6)
   NTraj  = 1
   
   dtE = dtN/600
   nsites  = 6001
   nmodes  = 301
   NStates = nsites + nmodes
   ndof    = nsites
   M = 250 * amu #* np.ones((nsites)) 
   initState = NStates//2
   nskip = 1
   #------------------  
   β  = 1052.8 # Temp
   αm, βm = 1500 /(cm * Å) , 3000/(cm * Å)
   τ  = 150/cm     
   K  = 14500.0 * (amu/ps**2)
   #-------------------
   dL   = 100 * Å
   L    = nsites * dL
   ωc0 = (1.65 + 0.57 - 0.2)/27.2114 
   kx  = ωc0/c
   j =  (np.arange(nmodes)) #- (nmodes-1)/2 
   kz  =   2 * π * (j-nmodes//2)/L  #
   #kz  =  π * (j + 1)/ (nsites * L)  
   ωc  = ( ωc0**2 + (kz * c)**2 )**0.5
   Δωc = 0.2/27.2114
   Ex = ωc0 + Δωc
   
   gc  = 0.15/27.2114/(nsites**0.5)
   
   E0 = -0.25/27.2114 # relative to the exciton branch Ex 
   dE = 0.05/27.2114

def Hel(R):
    αm  = parameters.αm 
    τ   = parameters.τ 
    ns  = parameters.NStates
    nx  = parameters.nsites
    Ex  = parameters.Ex
    ωc  = parameters.ωc
    kz  = parameters.kz
    dL   = parameters.dL
    Vij = np.zeros((ns,ns), dtype = 'complex') 
    gc  = parameters.gc
    βm  = parameters.βm 
    ωc0 = parameters.ωc0

    Vij[np.diag_indices(nx)] = βm * R[:]
    for i in range(nx-1):
        Vij[i+1,i] += αm * (R[i+1]-R[i]) - τ
        Vij[i,i+1] += αm * (R[i+1]-R[i]) - τ
    Vij[0,nx-1] += αm * (R[0]-R[nx-1]) - τ  
    Vij[nx-1,0] += αm * (R[0]-R[nx-1]) - τ  
    for i in range(nx, ns):
        Vij[i,i] += ωc[i-nx] - Ex
        for j in range(nx):
            rj = dL * (j+1)  
            #rj = L * j 
            θ  = np.arctan(kz[i-nx]/parameters.kx)
            #Vij[i,j] += gc * np.sin(kz[i-nx] * rj) #* np.cos(θ)
            #Vij[j,i] += gc * np.sin(kz[i-nx] * rj) #* np.cos(θ)
            Vij[i,j] += gc * np.exp( -1j * kz[i-nx] * rj) * np.cos(θ) * (ωc[i-nx]/ωc0)**0.5
            Vij[j,i] += gc * np.exp(  1j * kz[i-nx] * rj) * np.cos(θ) * (ωc[i-nx]/ωc0)**0.5
            #Vij[i,j] += gc * np.cos(kz[i-nx] * rj) #* np.cos(θ)
            #Vij[j,i] += gc * np.cos(kz[i-nx] * rj) #* np.cos(θ)

    return Vij
    
def dHel0(R):
    K = parameters.K
    return K * R

def dHel(R):
    param = parameters
    ns = param.NStates
    nx  = param.nsites
    αm = param.αm
    
    dVij = csr_matrix((ns,ns,len(R))) #np.zeros((ns,ns,len(R)))

    for i in range(nx-1):
        dVij[i+1,i,i+1], dVij[i+1,i,i+1] =  αm,  αm 
        dVij[i,i+1,i],   dVij[i+1,i,i]   = -αm, -αm  
    dVij[0,nx-1,0], dVij[nx-1,0,0]       =  αm,  αm 
    dVij[0,nx-1,nx-1], dVij[nx-1,0,nx-1] = -αm, -αm 
    
    return dVij 


def initR():
    R0 = 0.0
    P0 = 0.0
    param = parameters
    M  = param.M 
    K  = param.K 
    ndof = param.ndof
    β = param.β
    σP = (M/β)**0.5
    σR = (1/(β*K))**0.5
    R, P = np.zeros((ndof)), np.zeros((ndof))
    R = np.random.normal(size = (ndof))*σR + R0
    P = np.random.normal(size = (ndof))*σP + P0
    return R, P
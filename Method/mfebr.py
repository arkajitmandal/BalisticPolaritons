import numpy as np
from scipy.sparse import csr_matrix


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the electronic part
def initElectronic(Nstates, initState = 0):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    c = np.zeros((Nstates), dtype='complex128')
    c[initState] = 1.0
    return c


def localize(U, E, E0, dE, nsites):

    Ei = E[(E>E0) & (E< E0 + dE)] * 1
    Uj = U[:,(E>E0) & (E< E0 + dE)] * 1
    nSteps = 10000
    Ci = np.ones((len(Ei)))/ len(Ei)**0.5 + 0j
    print ("--------------------------------------------------")
    print (np.sum(np.abs(getΨ(Ci, Uj)[:nsites])**2.0))
    Ψ0 = np.abs(getΨ(Ci, Uj)[:nsites])**2
    
    
    for i in range(nSteps):

        C0 = Ci * 1.0
        θ  = np.random.random() * (2 * np.pi)
        jdx = np.random.randint(0,len(Ci))
        Ci[jdx] *=  np.exp(-1j * θ)        
        Ci = Ci/np.sum(Ci*Ci.conjugate())**0.5
        
        Ci = choose(Ci, C0, Uj, nsites)
         
        C0 = Ci * 1.0/np.sum(Ci*Ci.conjugate())
        
        jdx = np.random.randint(0,len(Ci))
        Ci[jdx] += (np.random.random() - 0.5) #* np.exp(-1j * θ)
        N  = np.sum(Ci*Ci.conjugate())
        Ci = (Ci)/N**0.5
 
        Ci = choose(Ci, C0, Uj, nsites)

    return getΨ(Ci, Uj) 

     

def getΨ(Ci, Ui):
    Ψi = np.zeros(len(Ui[:,0]))
    Ψi = Ui @ Ci
    #for i in range(len(Ψi)):
    #    Ψi[i] = np.sum(Ci[:] * Ui[i,:])
    return Ψi
    
"""
def DX2(Ci, Ui, nsites):
    Ψi = getΨ(Ci, Ui) 
    
    R = np.arange(0,len(Ψi)) - 3000
    R[nsites:] = 0
    return (np.sum((Ψi*Ψi.conjugate()) * R**2) - np.sum((Ψi*Ψi.conjugate()) * R)**2 ) / np.sum((np.abs(Ψi[:nsites])**2))
"""

def DX2(Ci, Ui, nsites):
    Ψi = getΨ(Ci, Ui) 
    

    pii = (Ψi*Ψi.conjugate()).real
    pii =  pii / np.sum(pii[:nsites])
 
    R = np.arange(0,len(Ψi))  
    R[nsites:] = 0
    
    #return (np.sum((pii) * R**2) - np.sum((pii) * R)**2 ) / np.sum(pii**2)
    A =  (np.sum(pii * R**2) - np.sum(pii * R)**2 ) #
    return A

def choose(C1, C2, Ui, nsites):
    dx1 = DX2(C1, Ui, nsites)
    dx2 = DX2(C2, Ui, nsites) 

    if dx1<dx2 : 
        #print ("change")
        #plt.plot(getΨ(C1, Ui)**2)
        return C1 
    return C2



#plt.ylim(0,0.1)


def propagateCi(ci,Vij, dt):
    c = ci * 1.0
    # https://thomasbronzwaer.wordpress.com/2016/10/15/numerical-quantum-mechanics-the-time-dependent-schrodinger-equation-ii/
    ck1 = (-1j) * (Vij @ c)
    ck2 = (-1j) * Vij @ (c + (dt/2.0) * ck1 )
    ck3 = (-1j) * Vij @ (c + (dt/2.0) * ck2 )
    ck4 = (-1j) * Vij @ (c + (dt) * ck3 )
    c = c + (dt/6.0) * (ck1 + 2.0 * ck2 + 2.0 * ck3 + ck4)
    return c


def propagateCii(ci,Vij, dt):
    Ei, Ui = np.linalg.eigh(Vij)
    c = Ui.T @ ci
    c = Ui @ (np.exp(-1j * Ei * dt) * c)
    return c

def Force(dat):

    #dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0  = dat.dH0 
    αm   = dat.param.αm
    βm   = dat.param.βm
    ci = dat.ci
    cj = ci.conjugate() 
    
    nsite = dat.param.nsites
    nmode = dat.param.nmodes


    F = -dH0 #np.zeros((len(dat.R)))
    for i in range(len(F)-1):
        F[i] -= βm  * (cj[i] * ci[i]).real 
        F[i] -= 2.0 * αm * ( - (cj[i+1] * ci[i]).real  + (cj[i-1] * ci[i]).real  )
    F[-1] -=    2.0 * αm * ( - (cj[0] * ci[nsite-1]).real  + (cj[nsite-2] * ci[nsite-1]).real  )

    return F#-dat.dH0 

def VelVer(dat) : 
    par =  dat.param
    v = dat.P/par.M
    F1 = dat.F1 
    # electronic wavefunction
    ci = dat.ci * 1.0
    
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep
    Vij = csr_matrix(dat.Hij)
    # half electronic evolution
    for t in range(int(np.floor(EStep/2))):
        ci = propagateCi(ci, Vij, dtE)  
    ci /= np.sum(ci.conjugate()*ci) 
    dat.ci = ci * 1.0 

    # ======= Nuclear Block ==================================
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R)
    Vij = csr_matrix(dat.Hij)
    #dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat) # force at t2
    v += 0.5 * (F1 + F2) * par.dtN / par.M
    dat.F1 = F2
    dat.P = v * par.M
    # ======================================================
    # half electronic evolution
    for t in range(int(np.ceil(EStep/2))):
        ci = propagateCi(ci, Vij, dtE)  
    ci /= np.sum(ci.conjugate()*ci)  
    dat.ci = ci * 1.0 

    return dat

def VelVerii(dat) : 
    par =  dat.param
    v = dat.P/par.M
    F1 = dat.F1 
    # electronic wavefunction
    
    
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep

    # half electronic evolution
    ci = dat.ci * 1.0
    dat.ci =  propagateCii(ci, dat.Hij, par.dtN/2)

    # ======= Nuclear Block ==================================
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R)
    #dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat) # force at t2
    v += 0.5 * (F1 + F2) * par.dtN / par.M
    dat.F1 = F2
    dat.P = v * par.M
    # ======================================================
    # half electronic evolution
    ci = dat.ci * 1.0
    dat.ci =  propagateCii(ci, dat.Hij, par.dtN/2)

    return dat

def dX(psi):
    ns = len(psi)
    R = np.arange(ns)
    dX = np.zeros((ns))
    for i in range(ns):
        psiR = np.roll(psi,i)
        dX[i] = np.sum(psiR*psiR.conjugate() * R**2) - np.sum(psiR * psiR.conjugate() * R) **2
    return np.min(dX)


def Ctheory(parameters):
    wk = parameters.ωc
    gc = parameters.gc * np.sqrt(parameters.nsites) 
    kz = parameters.kz 
 
    Ctheory = np.zeros((len(wk),2, 2))
    ek = parameters.Ex-parameters.τ * 2 
    for k in range(len(wk)):
        θ  = np.arctan(kz[k]/parameters.kx)
        H2x2 = np.zeros((2,2))
        H2x2[0,0] = ek  
        H2x2[1,1] = wk[k]
        H2x2[1,0], H2x2[0,1] = gc * np.cos(θ)**0.5 , gc * np.cos(θ)**0.5
        _, U = np.linalg.eigh(H2x2)
        Ctheory[k,:,:] = U[:,:] 
    return Ctheory


def Psik(Psi, eikn, parameters):
    N   =  parameters.nsites
    Ψk = Psi * 1
    # |en,0> ==> |ek,0>
    Ψk[:N] = eikn @  Psi[:N]
    return Ψk


def PsiPolk(psik, Ctheory, parameters):
    N   =  parameters.nsites
    nm  =  parameters.nmodes
 
    # We are going take a basis |-,k> and |+,k>
    # Note: |-,k> will run k=0, 6000 (6001 states)
    # But will have |+,k> will run k=0, 301
    # Now |ek,0> =  |-,k> for |k| > 150 
    # Final structure of the basis is :
    # |-, -3000> |-, -2999> ... |-, 0> ... |-, 3000> 
    # |+, -150>  |+, -149> ... |+, 0> ... |+, 149> |+, 150>

    U = np.identity((len(psik)))
    """
    for ik in range(nm):
        startIdx = N//2 - nm//2
        # <- k|ek 0>
        U[startIdx + ik, startIdx + ik] = Ctheory[ik, 0,0]
        # <- k|g 1k>
        U[startIdx + ik, N + ik] = Ctheory[ik, 1,0]
        # <+ k|g 1k>
        U[N + ik, N + ik] = Ctheory[ik, 1,1]
        # <+ k|ek 0> 
        U[N + ik, startIdx + ik] =  Ctheory[ik, 0,1]
    """
    startIdx = N//2 - nm//2
    
    k = np.arange(nm)
    # <- k|ek 0>
    U[(startIdx + k,startIdx + k)] = Ctheory[:, 0,0]
    # <- k|g 1k>
    U[(startIdx + k, N + k)]       = Ctheory[:, 1,0]
    # <+ k|g 1k>
    U[(N + k, N + k)]              = Ctheory[:, 1,1]
    # <+ k|ek 0> 
    U[(N + k, startIdx + k)]       = Ctheory[:, 0,1]

    return U @ psik

def PsiPoln(Polk, eikn, parameters):
    N   =  parameters.nsites
    nm  =  parameters.nmodes
    startIdx = N//2 - nm//2
 
    fullPsi = np.zeros((N*2)) + 0j 
    fullPsi[:N] =  Polk[:N]
    startIdx = N//2 - nm//2
    fullPsi[N+startIdx:N+startIdx+nm] = Polk[N:]
    eiknT = eikn.T.conjugate()
    fullPsi[:N] = eiknT @ fullPsi[:N]
    fullPsi[N:] = eiknT @ fullPsi[N:]
 
    return fullPsi 

def getPolariton(psi, Ctheory,eikn, parameters): 
    #----------------------------------
    psik   = Psik(psi, eikn, parameters)
    pPsik = PsiPolk(psik, Ctheory, parameters)
    pPsin = PsiPoln(pPsik, eikn, parameters)

    return pPsin


def pop(dat):
    ci =  dat.ci
    return np.outer(ci.conjugate(),ci)

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    nskip = parameters.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    #rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    X_ensemble   = np.zeros((parameters.nsites*2, NSteps//nskip + pl))
    wf_real = np.zeros((parameters.nsites*2, NSteps//nskip + pl))
    wf_imag = np.zeros((parameters.nsites*2, NSteps//nskip + pl))


    # Cthory
    Rot = Ctheory(parameters)
    # Eikn -----------------------------------
    N   =  parameters.nsites
    dL   = parameters.dL 
    j    =  (np.arange(N))  
    Rn   =  (1 + j) * dL 
    kz   =  2 * np.pi * (j-N//2)/( N * dL) 
    eikn = np.exp(-1j * np.einsum("i,j -> ij",  kz , Rn))/ N**0.5 
    #------------------------------------------

    # Ensemble
    for itraj in range(NTraj): 
        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()
        
        

        # set propagator
        vv  = VelVer

        # Call function to initialize mapping variables
        cD = initElectronic(NStates, initState) #np.random.randint(0,NStates)) # np.array([0,1])
        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        #dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        #----------------------------
        #Ei, Ui = np.linalg.eigh(dat.Hij)
        Ei, Ui = np.linalg.eigh(dat.Hij)
        E0 = dat.param.E0 - dat.param.dE/2 # making E0 the center #-0.25/27.2114
        dE = dat.param.dE # 0.05/27.2114
        """
        Uj = Ui[:,(Ei>E0) & (Ei< E0 + dE)] * 1
        modes = len(Uj[0,:])
        psiN = Uj[:,0] * 1 / np.sqrt(modes)
        for i in range(1,modes):
            psiN +=  Uj[:,i]  /  np.sqrt(modes)
        dat.ci = psiN #Ui @ cD
        """

        dat.ci = localize(Ui, Ei, E0, dE, parameters.nsites)
        #----------------------------
        #----------------------------
        dat.F1 = Force(dat) # Initial Force

        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                ns = parameters.nsites
                #ρij = pop(dat)[:ns,:ns]
                LP = getPolariton(dat.ci, Rot, eikn, parameters)
                #X_ensemble[0,iskip] += np.sum(psi*psi.conjugate() * r**2) - np.sum(psi*psi.conjugate() * r) **2
                wf_real[:,iskip] = LP.real 
                wf_imag[:,iskip] = LP.imag 
                # Excitonic Distribution
                #X_ensemble[0:ns,iskip] += (psi*psi.conjugate()).real
                # Polaritonic  Distribution
                X_ensemble[:,iskip]  += np.abs(LP)**2
                
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)

    return X_ensemble, wf_real, wf_imag

if __name__ == "__main__": 
    import spinBoson as model
    par =  model.parameters
    
    rho_ensemble = runTraj(par)
    
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates

    PiiFile = open("Pii.txt","w") 
    for t in range(NSteps):
        PiiFile.write(f"{t * model.parameters.nskip} \t")
        for i in range(NStates):
            PiiFile.write(str(rho_ensemble[i,i,t].real / NTraj) + "\t")
        PiiFile.write("\n")
    PiiFile.close()


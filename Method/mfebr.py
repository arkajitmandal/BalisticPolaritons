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
    ci[par.nsites:] = ci[par.nsites:] * np.exp(- par.Γc * par.dtN/4.0)
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
    ci[par.nsites:] = ci[par.nsites:] * np.exp(-par.Γc * par.dtN/4.0)
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
    X_ensemble   = np.zeros((parameters.nsites+1, NSteps//nskip + pl))
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
        E0 = -0.275/27.2114
        dE =  0.05/27.2114
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
                r  =  np.diag(np.arange(ns))   
                psi = dat.ci[:ns]
                X_ensemble[0, iskip] +=  np.sum((dat.ci[:]*dat.ci[:].conjugate()).real) #np.sum(psi*psi.conjugate() * r**2) - np.sum(psi*psi.conjugate() * r) **2
                X_ensemble[1:,iskip] += (psi*psi.conjugate()).real
                #rho_ensemble[:,:,iskip] += pop(dat)
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)

    return X_ensemble

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


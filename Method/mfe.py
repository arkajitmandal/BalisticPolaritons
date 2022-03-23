import numpy as np


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the electronic part
def initElectronic(Nstates, initState = 0):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    c = np.zeros((Nstates), dtype='complex128')
    c[initState] = 1.0
    return c

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

    dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0  = dat.dH0 

    ci = dat.ci

    F = -dH0 #np.zeros((len(dat.R)))
    for i in range(len(ci)):
        F -= dH[i,i,:]  * (ci[i] * ci[i].conjugate() ).real
        for j in range(i+1, len(ci)):
            F -= 2.0 * dH[i,j,:]  * (ci[i].conjugate() * ci[j] ).real
    return F

def VelVer(dat) : 
    par =  dat.param
    v = dat.P/par.M
    F1 = dat.F1 
    # electronic wavefunction
    ci = dat.ci * 1.0
    
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep

    # half electronic evolution
    for t in range(int(np.floor(EStep/2))):
        ci = propagateCi(ci, dat.Hij, dtE)  
    ci /= np.sum(ci.conjugate()*ci) 
    dat.ci = ci * 1.0 

    # ======= Nuclear Block ==================================
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R)
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat) # force at t2
    v += 0.5 * (F1 + F2) * par.dtN / par.M
    dat.F1 = F2
    dat.P = v * par.M
    # ======================================================
    # half electronic evolution
    for t in range(int(np.ceil(EStep/2))):
        ci = propagateCi(ci, dat.Hij, dtE)  
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
    dat.dHij = par.dHel(dat.R)
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
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    X_ensemble   = np.zeros((NSteps//nskip + pl))
    # Ensemble
    for itraj in range(NTraj): 
        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()
        
        # set propagator
        vv  = VelVerii

        # Call function to initialize mapping variables
        cD = initElectronic(NStates, initState)#np.random.randint(0,NStates)) # np.array([0,1])
        
        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        #----------------------------
        Ei, Ui = np.linalg.eigh(dat.Hij)
        E0 = -0.5/27.2114
        dE =  0.05/27.2114
        Uj = Ui[:,(Ei>E0) & (Ei< E0 + dE)] * 1
        modes = len(Uj[0,:])
        psiN = Uj[:,0] * 1 / np.sqrt(modes)
        for i in range(1,modes):
            psiN +=  Uj[:,i]  /  np.sqrt(modes)
        dat.ci = psiN #Ui @ cD
        #----------------------------
        dat.F1 = Force(dat) # Initial Force

        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                """
                ρij = pop(dat)
                r2 =  np.diag(np.arange(len(ρij))**2)  
                r  =  np.diag(np.arange(len(ρij)))   

                r2, r =  np.trace(ρij @ r2), np.trace(ρij @ r)
                X_ensemble[iskip] += r2 - r**2
                #rho_ensemble[:,:,iskip] += pop(dat)
                """
                ns = parameters.nsites
                #ρij = pop(dat)[:ns,:ns]
                r  =  np.diag(np.arange(ns))   
                psi = dat.ci[:ns] * 1
                X_ensemble[iskip] += np.sum(psi*psi.conjugate() * r**2) - np.sum(psi*psi.conjugate() * r) **2
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


#!/usr/bin/env python3
#SBATCH -o output.log
#SBATCH -t 1:00:00

import sys, os

sys.path.append(os.popen("pwd").read().replace("\n","")+"/Method")
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")
#-------------------------
try:
    input = open(sys.argv[1], 'r').readlines()
    print(f"Reading {sys.argv[1]}")
except:
    print("Reading input.txt")
    input = open('input.txt', 'r').readlines()

def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")

model_ =  getInput(input,"Model")
method_ = getInput(input,"Method").split("-")
exec(f"import {model_} as model")
exec(f"import {method_[0]} as method")
try:
    stype = method_[1]
except:
    stype = "_"
#-------------------------
from multiprocessing import Pool
import time 
import numpy as np

t0 = time.time()
#----------------
trajs = model.parameters.NTraj
#----------------
try:
    fold = sys.argv[2]
    
except:
    fold = "."
print (f"Folder: {fold}")
#------------------------------------------------------------------------------------------
try:
    procs = int(getInput(input,"Cpus"))
except:
    procs = 1
ntraj = procs * trajs
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

t1 = time.time()
with Pool(procs) as p:

    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates

    #------ Arguments for each CPU------------------
    args = []
    for j in range(procs):
        par = model.parameters() 
        par.ID   = j
        par.SEED   = np.random.randint(0,100000000)
        
        #---- methods in model ------

        par.dHel = model.dHel
        par.dHel0 = model.dHel0
        par.initR = model.initR
        par.Hel   = model.Hel
        if method_[0]=="nrpmd":
            par.initHel0 = model.initHel0
        par.stype = stype
        args.append(par)
    #-----------------------------------------------
    print(f"Running : {par.NTraj*procs}  trajectories in {procs} cpu" )

    #------------------- parallelization -----------------------------------
    rho_ensemble  = p.map(method.runTraj, args)
    #-----------------------------------------------------------------------

#------------------- Gather --------------------------------------------
rho_sum = np.zeros(rho_ensemble[0].shape, dtype = rho_ensemble[0].dtype)
for i in range(procs):
    for t in range(rho_ensemble[0].shape[-1]):
        rho_sum[:,:,t] += rho_ensemble[i][:,:,t]

try:    
    PiiFile = open(f"{fold}/{method_[0]}-{method_[1]}-{model_}.txt","w") 
except:
    PiiFile = open(f"{fold}/{method_[0]}-{model_}.txt","w") 
 
NTraj = model.parameters().NTraj
for t in range(rho_ensemble[0].shape[-1]):
    PiiFile.write(f"{t * model.parameters.nskip * model.parameters.dtN} \t")
    for i in range(NStates):
        PiiFile.write(str(rho_sum[i,i,t].real / (  procs * NTraj ) ) + "\t")
    PiiFile.write("\n")
PiiFile.close()
t2 = time.time()-t1
print(f"Total Time: {t2}")
print(f"Time per trajectory: {t2/ntraj}")
import numpy as np
import h5py
from netCDF4 import Dataset
import subprocess
import glob
import os
import sys
from time import time
from pathlib import Path
from datetime import date
import shutil

current_path=os.getcwd()

# Set parameter values
Ne = 20 # Ensemble size (including reference/truth)
ind_ref = 2 # Index of the reference/truth within the 80-member ensemble
Nt = 365 # Number of cycles (days)
No = 10 # Number of satellite obs per analysis cycle
obs_lag = 0 # Time shift for obs cycle

# get indices of the ensemble
ind_ens = list(range(1,Ne+1))
ind_ens.remove(ind_ref)

# Set up paths to executables & restart.nc files
base_dir = current_path+"/"
# Initialize random number generator
rng = np.random.default_rng(seed=0) # to ensure reproducibility

# Allocate space for the ensemble
aicen_ens_prior = np.zeros((Ne-1,5))
vicen_ens_prior = np.zeros((Ne-1,5))
vsnon_ens_prior = np.zeros((Ne-1,5))
aicen_ens_posterior = np.zeros((Ne-1,5))
vicen_ens_posterior = np.zeros((Ne-1,5))
vsnon_ens_posterior = np.zeros((Ne-1,5))

# restart file starts on JAN2
# check for date of last restart file
start = 0
with Dataset(base_dir + "mem0001/restart/forecast.nc", "r", format="NETCDF4") as restart_file:
    start = int( restart_file.istep1 / 24 ) - 1

# Kate addition: add date to the file
today = date.today()
output_filename = "output_No{}_Ne{}_ol{}_".format(No,Ne,obs_lag)
output_filename = output_filename+str(today)+".h5"

if start == 0:
    # Set up hdf5 file for output
    with h5py.File(base_dir+output_filename, "w") as da_out:
        da_out.create_dataset("forecast/aicen",(Nt,Ne-1,5))
        da_out.create_dataset("forecast/vicen",(Nt,Ne-1,5))
        da_out.create_dataset("forecast/vsnon",(Nt,Ne-1,5))
        da_out.create_dataset("analysis/aicen",(Nt,Ne-1,5))
        da_out.create_dataset("analysis/vicen",(Nt,Ne-1,5))
        da_out.create_dataset("analysis/vsnon",(Nt,Ne-1,5))
        da_out.create_dataset("ref/aicen",(Nt,5))
        da_out.create_dataset("ref/vicen",(Nt,5))
        da_out.create_dataset("ref/vsnon",(Nt,5))
        da_out.create_dataset("observations/No",(1))
        da_out.create_dataset("observations/k",(Nt))
        da_out.create_dataset("ess",(Nt))
        da_out["observations/No"][0] = No
else:
    assert (Path(base_dir)/output_filename).exists()

first_assimilation = 0
w = np.full(Ne-1, np.nan)
# Assimilation loop: Assimilate then forecast
for t in range(start, Nt):
    # Print cycle number
    print()
    print('------------------------- Cycle ',t,' -------------------------')
    tic_total = time()

    # Read the ref/true state
    with Dataset(base_dir+"mem"+f'{ind_ref:04}'+"/restart/forecast.nc", "r", format="NETCDF4") as rootgrp_ref:
        aicen_ref = np.array(rootgrp_ref["aicen"][:,2]) # Icepack runs 4 simulations; we want #3
        vicen_ref = np.array(rootgrp_ref["vicen"][:,2])
        vsnon_ref = np.array(rootgrp_ref["vsnon"][:,2])

    # Write ref/true state to output file
    with h5py.File(base_dir+output_filename, "r+") as da_out:
        da_out["ref/aicen"][t,:] = aicen_ref
        da_out["ref/vicen"][t,:] = vicen_ref
        da_out["ref/vsnon"][t,:] = vsnon_ref

    # Create synthetic observations: binomial sampling
    open_frac = 1. - np.sum(aicen_ref)  # fraction of open water in the reference sample
    k = rng.binomial(No,open_frac) # No total observations, k of them have open water

    # Read ensemble forecast
    for i in range(Ne-1):
        with Dataset(base_dir+"mem"+f'{ind_ens[i]:04}'+"/restart/forecast.nc", "r", format="NETCDF4") as rootgrp_ens:
            aicen_ens_prior[i,:] = np.array(rootgrp_ens["aicen"][:,2])
            vicen_ens_prior[i,:] = np.array(rootgrp_ens["vicen"][:,2])
            vsnon_ens_prior[i,:] = np.array(rootgrp_ens["vsnon"][:,2])

    # Write forecast ensemble and synthetic observations to output file
    with h5py.File(base_dir+output_filename, "r+") as da_out:
        da_out["forecast/aicen"][t,:,:] = aicen_ens_prior
        da_out["forecast/vicen"][t,:,:] = vicen_ens_prior
        da_out["forecast/vsnon"][t,:,:] = vsnon_ens_prior
        da_out["observations/k"][t] = k

    # SIR Particle Filter
    tic_assimilation = time()
    
    if (not (t-first_assimilation)%5):
        #   PF: Get open water fractions forcast ensemble
        a0_prior = 1. - np.sum(aicen_ens_prior,axis=1)

        #   PF: Compute weights
        w = a0_prior**k * (1 - a0_prior)**(No - k)
        w = np.exp(np.log(w) - np.max(np.log(w)))
        w /= w.sum()
        print(k, No, a0_prior)
        print(w)
        print('Effective sample size is {}'.format(1/np.sum(w**2)))

        #   PF: Multinomial resampling
        ind_resampled = np.array(ind_ens)
        for n in range(Ne-1):
            ind = np.random.choice(Ne - 1, p=w)
            ind_resampled[n] = ind_ens[ind]
            aicen_ens_posterior[n,:] = aicen_ens_prior[ind,:]
            vicen_ens_posterior[n,:] = vicen_ens_prior[ind,:]
            vsnon_ens_posterior[n,:] = vsnon_ens_prior[ind,:]
    else: 
        ind_resampled = np.array(ind_ens)
        aicen_ens_posterior = np.array(aicen_ens_prior)  # Using np.array() creates a deepcopy
        vicen_ens_posterior = np.array(vicen_ens_prior)
        vsnon_ens_posterior = np.array(vsnon_ens_prior)

    # Done with assimilation
    toc_assimilation = time()

    # Write analysis ensemble to output file
    with h5py.File(base_dir+output_filename, "r+") as da_out:
        da_out["analysis/aicen"][t,:,:] = aicen_ens_posterior
        da_out["analysis/vicen"][t,:,:] = vicen_ens_posterior
        da_out["analysis/vsnon"][t,:,:] = vsnon_ens_posterior
        da_out["ess"][t] = 1/np.sum(w**2)

    # Copy restarts
    for i in range(Ne-1):
        shutil.copyfile(base_dir+"mem"+f'{ind_resampled[i]:04}'+"/restart/forecast.nc",
            base_dir+"mem"+f'{ind_ens[i]:04}'+"/restart/analysis.nc")

    # Run forecasts
    tic_forecast = time()
    with open('forecast_err.txt', 'a') as err_file:
        subprocess.run(base_dir+"run_all.csh",cwd=base_dir, stdout=subprocess.DEVNULL, stderr=err_file)
    # Check if all forecasts completed successfully
    for i in range(1,Ne+1):
        run_dir = base_dir+"mem"+f'{i:04}'+"/"
        if not glob.glob(run_dir+'restart/restart.nc'):
            print('Something went wrong with member',i)
            exit()
    toc_forecast = time()

    toc_total = time()

    print('time: ')
    print('\tassimilation: ', toc_assimilation - tic_assimilation)
    print('\tforecast:           ', toc_forecast - tic_forecast)
    print('\ttotal:              ', toc_total - tic_total)
    sys.stdout.flush()
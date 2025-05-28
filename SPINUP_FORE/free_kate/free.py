import numpy as np
import h5py
from netCDF4 import Dataset
import subprocess
import glob
import os
import logging
from scipy.stats import beta
import sys
import copy
from time import time
from pathlib import Path
# New from Kate- adding to have dates appear on output files
from datetime import datetime

# Values of salinity to use to initialize ice that appears during the analysis cycle
s_init = np.array([[5.9, 3.6, 2.6, 2.3, 2.4, 2.8, 5.6],
                   [3.0, 2.6, 2.6, 2.5, 2.5, 2.8, 4.5],
                   [1.6, 1.5, 1.7, 2.0, 2.2, 2.8, 3.7],
                   [.12, .19, .31, .46, .68, 1.1, 3.5],
                   [.01, .18, .31, .48, .67, .97, 1.5]])

current_path=os.getcwd()

# Set parameter values
Ne = 40 # Ensemble size (including reference/truth)
ind_ref = 2 # Index of the reference/truth within the 80-member ensemble
Nt = 3285 # Number of cycles (days)
No = 10 # Number of satellite obs
obs_lag = 0 # Time shift for obs cycle

# get indices of the ensemble
ind_ens = list(range(1,Ne+1))
ind_ens.remove(ind_ref)

# bounds that define thickness categories
h_bnd = np.array([[0.00001               ,0.6445072168194257],
                  [0.6445072168194258  ,1.3914334975763035],
                  [1.3914334975763037  ,2.470179381959888],
                  [2.470179381959889   ,4.567287918850491],
                  [4.567287918850492   ,9.3338418158681743]])

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

# Kate's filename
now = datetime.now()
dt_string = now.strftime("%d_%m_%H:%M")
output_filename = "output_No{}_Ne{}_ol{}_".format(No,Ne,obs_lag)
output_filename = output_filename+dt_string+".h5"

#output_filename = "output_No{}_Ne{}_ol{}.h5".format(No,Ne,obs_lag)  #Ian's filename

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
        da_out["observations/No"][0] = No
else:
    assert (Path(base_dir)/output_filename).exists()


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
    obs_info = np.array([No, k],dtype=np.float64)
    obs_info.tofile("obs_info.dat")

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

    # Convert from snow volume to snow thickness
    hsnon_ens = np.nan_to_num(vsnon_ens_prior/aicen_ens_prior)

    tic_simplex = time()
    if( (t-obs_lag)%400 ):
        aicen_ens_posterior = aicen_ens_prior
        vicen_ens_posterior = vicen_ens_prior
        vsnon_ens_posterior = vsnon_ens_prior
    else:
        # As described in Grooms & Riedel, assimilation should happen in this block every 10 days.
        # But since it's a free run we'll just set analysis=forecast.
        aicen_ens_posterior = aicen_ens_prior
        vicen_ens_posterior = vicen_ens_prior
        vsnon_ens_posterior = vsnon_ens_prior

    # Done with assimilation
    toc_simplex = time()

    # Write analysis ensemble to output file
    with h5py.File(base_dir+output_filename, "r+") as da_out:
        da_out["analysis/aicen"][t,:,:] = aicen_ens_posterior
        da_out["analysis/vicen"][t,:,:] = vicen_ens_posterior
        da_out["analysis/vsnon"][t,:,:] = vsnon_ens_posterior

    # Write restarts
    for i in range(Ne-1):
        # print('writing restart for member ', i)
        with Dataset(base_dir+"mem"+f'{ind_ens[i]:04}'+"/restart/analysis.nc", "r+", format="NETCDF4") as rootgrp_ens:
            rootgrp_ens["aicen"][:,2] = aicen_ens_posterior[i,:] 
            rootgrp_ens["vicen"][:,2] = vicen_ens_posterior[i,:] 
            rootgrp_ens["vsnon"][:,2] = vsnon_ens_posterior[i,:]
            # Fix enthalpy for ice that has been zeroed out
            for j in range(5):
                if ((aicen_ens_prior[i,j] > 0.0) and (aicen_ens_posterior[i,j] == 0.0)):
                    for var in rootgrp_ens.variables.keys():
                        if var.startswith("qice"):
                            rootgrp_ens[var][j,2] = 0.0
                        if var.startswith("qsno"):
                            rootgrp_ens[var][j,2] = 0.0
            # Fix enthalpy for ice that grew from zero
            for j in range(5):
                if ((aicen_ens_prior[i,j] == 0.0) and (aicen_ens_posterior[i,j] > 0.0)):
                    for var in rootgrp_ens.variables.keys():
                        if var.startswith("qice"):
                            rootgrp_ens[var][j,2] = -3.e-8
            # Fix salinity for ice that grew from zero
            for j in range(5):
                if ((aicen_ens_prior[i,j] == 0.0) and (aicen_ens_posterior[i,j] > 0.0)):
                    for k in range(1,8):
                        var = "sice"+f'{k:03}'
                        rootgrp_ens[var][:,2] = s_init[:,k-1]
            
           # Fix zero salinity when zero area/volume
            for j in range(5):
               if (aicen_ens_posterior[i,j] == 0.0):
                      for k in range(1,8):
                          var = "sice"+f'{k:03}'
                          rootgrp_ens[var][j,2] = 0
            
            # Fix variables that need to be zero if aicen = 0
            var = ["FY", "alvl", "vlvl", "aerosnossl001", "aerosnoint001", "aeroicessl001", "aeroiceint001"]
            for j in range(5):
                 if (aicen_ens_posterior[i,j] == 0.0):
                         for index in range(len(var)):
                              rootgrp_ens[var[index]][j,2] = 0
                              #print('Zeroed variable '+ var[index] + ' in category ' + str(j))
                         



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
    print('\tsimplex assimilate: ', toc_simplex - tic_simplex)
    print('\tforecast:           ', toc_forecast - tic_forecast)
    print('\ttotal:              ', toc_total - tic_total)
    sys.stdout.flush()

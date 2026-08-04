# SeaIce

Files for running [Icepack](https://github.com/CICE-Consortium/Icepack) with data assimilation.

## Overview

This repository contains forcing data, initial conditions, configuration scripts, and code for running the Icepack sea ice column
physics model with various data assimilation algorithms. There are algorithms for assimilating Sea Ice Concentration (SIC) and Sea Ice Thickness (SIT).

## Repository Structure

```
SeaIce/
├── FORCING_FILES/        # Atmospheric/oceanic forcing data used to drive Icepack runs
    └── CAM6
    └── IPSOL
├── SPINUP_FORE/           # Icepack-DA configurations, algorithms, and forecasting 
    └── 2step/             # Novel SEnT algorithm for SIC assimilation
    └── 2step_simple/      # Simplified Scaling for 2nd step SIC assimilation
    └── Analyze Data/      # Helpful scripts for analyzing DA output
    └── EnKF/              # SIC assimilation
    └── Optimization_L2/   # 2 Step SIT assimilation using L2 optimization for the 2nd step
    └── Particle_filter/   # SIC assimilation
    └── free_kate/         # No assimilation
    └── initial_conditions/
```
Each one of the Icepack-DA folders within SPINUP_FORE requires the following 
```
     ├── input.nml           # Namelist configuration for DART (used for step one)
     ├── initialize.csh      # Sets up a run directory / initial conditions for each Icepack experiment
     ├── setup.csh           # Directory setup script
     ├── run_all.csh         # Top-level driver script that runs Icepack after DA
     └── icepack*            # Icepack configuration files
```


## Prerequisites

- An `icepack` conda environment (see `environment.yml` or set up manually — add setup instructions here)
- [Icepack](https://github.com/CICE-Consortium/Icepack) built and available
- If using a DART executable for Step One
      - [DART](https://github.com/NCAR/DART) built with the necessary executable compiled and in the appropriate run directory
- Python 3 with the packages required by `ITD_funcs.py` and `optimization_L2.py`

## Usage

1. Activate the environment:
   ```
   conda activate icepack
   ```
2. Set up the run directory:
   ```
   cd Optimization_L2
   ./setup.csh
   ./initialize.csh
   ```
3. Run the assimilation script (which calls Icepack):
   ```
   python Optimization_L2.py
   ```


## Author

Kate Boden

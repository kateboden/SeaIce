#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Figures
Created: 5/8/25

@author: kabo1917
"""

import dataOutput as da
import finalFiguresFunctions as ff
import numpy as np
from matplotlib import pyplot as plt
import h5py

# Global Variables for plots
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'



# Transport Files
filesT = ff.loadFiles(("../2step/output_No10_Ne20_ol0_2024-11-08.h5", "../2step/output_No10_Ne20_ol0_2024-12-04_1.h5", "../2step/output_No10_Ne20_ol0_2024-12-04_2.h5", "../2step/output_No10_Ne20_ol0_2024-12-04_3.h5", "../2step/output_No10_Ne20_ol0_2024-12-04_4.h5"), 
                   ("../2step/output_No10_Ne40_ol0_2024-11-11.h5", "../2step/output_No10_Ne40_ol0_2024-12-04_1.h5", "../2step/output_No10_Ne40_ol0_2024-12-04_2.h5", "../2step/output_No10_Ne40_ol0_2024-12-04_3.h5", "../2step/output_No10_Ne40_ol0_2024-12-04_4.h5"),
                   ("../2step/output_No10_Ne60_ol0_2024-11-12.h5", "../2step/output_No10_Ne60_ol0_2024-12-05_1.h5", "../2step/output_No10_Ne60_ol0_2024-12-05_2.h5", "../2step/output_No10_Ne60_ol0_2024-12-05_3.h5", "../2step/output_No10_Ne60_ol0_2024-12-05_4.h5"),
                   ("../2step/output_No10_Ne80_ol0_2024-11-11.h5", "../2step/output_No10_Ne80_ol0_2024-12-04_1.h5", "../2step/output_No10_Ne80_ol0_2024-12-04_2.h5", "../2step/output_No10_Ne80_ol0_2024-12-04_3.h5", "../2step/output_No10_Ne80_ol0_2024-12-04_4.h5"))
# EnKF Files
filesE = ff.loadFiles(("../EnKF/output_No10_Ne20_ol0_2025-01-03_0.h5","../EnKF/output_No10_Ne20_ol0_2025-01-03_1.h5", "../EnKF/output_No10_Ne20_ol0_2025-01-03_2.h5", "../EnKF/output_No10_Ne20_ol0_2025-01-03_3.h5", "../EnKF/output_No10_Ne20_ol0_2025-01-03_4.h5" ),
                   ("../EnKF/output_No10_Ne40_ol0_2024-11-11.h5","../EnKF/output_No10_Ne40_ol0_2025-01-03_1.h5", "../EnKF/output_No10_Ne40_ol0_2025-01-03_2.h5", "../EnKF/output_No10_Ne40_ol0_2025-01-03_3.h5", "../EnKF/output_No10_Ne40_ol0_2025-01-03_4.h5" ),
                   ("../EnKF/output_No10_Ne60_ol0_2024-11-13.h5","../EnKF/output_No10_Ne60_ol0_2024-12-07_1.h5", "../EnKF/output_No10_Ne60_ol0_2024-12-07_2.h5", "../EnKF/output_No10_Ne60_ol0_2024-12-07_3.h5", "../EnKF/output_No10_Ne60_ol0_2024-12-07_4.h5" ),
                   ("../EnKF/output_No10_Ne80_ol0_2024-11-12.h5","../EnKF/output_No10_Ne80_ol0_2024-12-07_1.h5", "../EnKF/output_No10_Ne80_ol0_2024-12-07_2.h5", "../EnKF/output_No10_Ne80_ol0_2024-12-07_3.h5", "../EnKF/output_No10_Ne80_ol0_2024-12-07_4.h5" ))
# Scale Files
filesS = ff.loadFiles(("../2step_simple/output_No10_Ne20_ol0_2024-11-08.h5", "../2step_simple/output_No10_Ne20_ol0_2024-12-07_1.h5", "../2step_simple/output_No10_Ne20_ol0_2024-12-07_2.h5", "../2step_simple/output_No10_Ne20_ol0_2024-12-07_3.h5", "../2step_simple/output_No10_Ne20_ol0_2024-12-07_4.h5"),
                   ("../2step_simple/output_No10_Ne40_ol0_2024-11-12.h5", "../2step_simple/output_No10_Ne40_ol0_2024-12-07_1.h5", "../2step_simple/output_No10_Ne40_ol0_2024-12-07_2.h5", "../2step_simple/output_No10_Ne40_ol0_2024-12-07_3.h5", "../2step_simple/output_No10_Ne40_ol0_2024-12-07_4.h5"),
                   ("../2step_simple/output_No10_Ne60_ol0_2024-11-14.h5", "../2step_simple/output_No10_Ne60_ol0_2024-12-06_1.h5", "../2step_simple/output_No10_Ne60_ol0_2024-12-06_2.h5", "../2step_simple/output_No10_Ne60_ol0_2024-12-06_3.h5", "../2step_simple/output_No10_Ne60_ol0_2024-12-06_4.h5"),
                   ("../2step_simple/output_No10_Ne80_ol0_2024-11-13.h5", "../2step_simple/output_No10_Ne80_ol0_2024-12-05_1.h5", "../2step_simple/output_No10_Ne80_ol0_2024-12-05_2.h5", "../2step_simple/output_No10_Ne80_ol0_2024-12-06_3.h5", "../2step_simple/output_No10_Ne80_ol0_2024-12-06_4.h5"))
# Particle Filter Files
filesP = ff.loadFiles(("../Particle_filter/output_No10_Ne20_ol0_2025-02-03_0.h5", "../Particle_filter/output_No10_Ne20_ol0_2025-02-03_1.h5","../Particle_filter/output_No10_Ne20_ol0_2025-02-03_2.h5","../Particle_filter/output_No10_Ne20_ol0_2025-02-03_3.h5", "../Particle_filter/output_No10_Ne20_ol0_2025-02-03_4.h5"),
                      ("../Particle_filter/output_No10_Ne40_ol0_2025-01-31_0.h5", "../Particle_filter/output_No10_Ne40_ol0_2025-01-31_1.h5","../Particle_filter/output_No10_Ne40_ol0_2025-01-31_2.h5","../Particle_filter/output_No10_Ne40_ol0_2025-01-31_3.h5", "../Particle_filter/output_No10_Ne40_ol0_2025-01-31_4.h5"),
                      ("../Particle_filter/output_No10_Ne60_ol0_2025-01-31_0.h5", "../Particle_filter/output_No10_Ne60_ol0_2025-01-31_1.h5","../Particle_filter/output_No10_Ne60_ol0_2025-01-31_2.h5","../Particle_filter/output_No10_Ne60_ol0_2025-01-31_3.h5", "../Particle_filter/output_No10_Ne60_ol0_2025-01-30_4.h5"),
                      ("../Particle_filter/output_No10_Ne80_ol0_2025-01-27_0.h5", "../Particle_filter/output_No10_Ne80_ol0_2025-01-30_1.h5","../Particle_filter/output_No10_Ne80_ol0_2025-01-30_2.h5","../Particle_filter/output_No10_Ne80_ol0_2025-01-30_3.h5", "../Particle_filter/output_No10_Ne80_ol0_2025-01-30_4.h5"))


var = 'aicen'
# No Assimilation
free = da.dataOutput("../free_kate/output_No10_Ne80_ol0_10_11.h5" , var) 
free = [free, free, free, free, free] 

#%%
"""
AEM subplot figure

"""
# Global Variables for plots
plt.rcParams['font.size'] = 40
plt.rcParams['font.family'] = 'Times New Roman'


import finalFiguresFunctions as ff
fig, axs = plt.subplots(2, 2, figsize=(20, 14))
plt.subplots_adjust(wspace=0.05)
row = [0, 0, 1, 1]
col = [0, 1, 0, 1]
for i in range(4):
    transport = ff.combineDatasets(filesT[i], var)
    enkf = ff.combineDatasets(filesE[i], var)
    scale = ff.combineDatasets(filesS[i], var)
    particle = ff.combineDatasets(filesP[i], var)
    ff.AEMfigure(transport, enkf, scale, free, particle, axs[row[i],col[i]])

# Add legend to first subplot
axs[0,0].legend(['Free', 'ENKF', 'SEnT', 'Scale', 'PF'], loc = 'upper right', fontsize = 24)

# Set y-ticks
axs[0,0].set_yticks(np.arange(0,0.1,0.03))
axs[0,1].set_yticks(np.arange(0,0.1,0.03))
axs[1,0].set_yticks(np.arange(0,0.08,0.02))
axs[1,1].set_yticks(np.arange(0,0.08,0.02))

# Set individual titles
axs[0,0].set_title('N$_e$ = 20')
axs[0,1].set_title('N$_e$ = 40')
axs[1,0].set_title('N$_e$ = 60')
axs[1,1].set_title('N$_e$ = 80')


# Remove x-ticks for top row
axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])

# Remove y-ticks second column
axs[0,1].set_yticklabels([])
axs[1,1].set_yticklabels([])

# Add labels
fig.supylabel('AEM')
fig.supxlabel('Ice Categories', fontsize = 45)

fig.tight_layout()

# Save Figure
plt.savefig('Figures_final/AEM_FINAL.png', dpi = 360)

#%%
"""
CRPS subplot figure

"""
# Global Variables for plots
plt.rcParams['font.size'] = 40
plt.rcParams['font.family'] = 'Times New Roman'


import finalFiguresFunctions as ff

fig, axs = plt.subplots(2, 2, figsize=(20, 14), sharex = True)
plt.subplots_adjust(wspace=0.05)
row = [0, 0, 1, 1]
col = [0, 1, 0, 1]
for i in range(4):
    transport = ff.combineDatasets(filesT[i], var)
    enkf = ff.combineDatasets(filesE[i], var)
    scale = ff.combineDatasets(filesS[i], var)
    particle = ff.combineDatasets(filesP[i], var)
    ff.CRPSfigure(transport, enkf, scale, free, particle, axs[row[i],col[i]])

# Add legend to first subplot
axs[0,0].legend(['Free', 'ENKF', 'SeNT', 'Scale', 'PF'], loc = 'upper right', fontsize = 24)

# Set y-ticks
axs[0,0].set_yticks(np.arange(0,0.065,0.02))
axs[0,1].set_yticks(np.arange(0,0.065,0.02))
axs[1,0].set_yticks(np.arange(0,0.035,0.01))
axs[1,1].set_yticks(np.arange(0,0.035,0.01))

# Set individual titles
axs[0,0].set_title('N$_e$ = 20')
axs[0,1].set_title('N$_e$ = 40')
axs[1,0].set_title('N$_e$ = 60')
axs[1,1].set_title('N$_e$ = 80')


# Remove x-ticks for top row
axs[0,0].set_xticklabels([])
axs[0,1].set_xticklabels([])
axs[1,1].set_xticks([0,1,2,3,4,5])
axs[1,1].set_xticklabels([0,1,2,3,4,5])

# Remove y-ticks second column
axs[0,1].set_yticklabels([])
axs[1,1].set_yticklabels([])



# Add labels
fig.supylabel('CRPS')
fig.supxlabel('Ice Categories', fontsize = 45)

fig.tight_layout()

# Save Figure
plt.savefig('Figures_final/CRPS_FINAL.png', dpi = 360)

#%%

"""
Bias figure

"""

# Global Variables for plots
plt.rcParams['font.size'] = 32
plt.rcParams['font.family'] = 'Times New Roman'


import finalFiguresFunctions as ff

i = 3  # Select ensemble size index 3 corresponds to 80 ensemble members
transport = ff.combineDatasets(filesT[i], var)
enkf = ff.combineDatasets(filesE[i], var)
scale = ff.combineDatasets(filesS[i], var)
particle = ff.combineDatasets(filesP[i], var)

tBias = ff.combineBias(transport)
eBias = ff.combineBias(enkf)
sBias = ff.combineBias(scale)
fBias = ff.combineBias(free)
pBias = ff.combineBias(particle)

t = np.arange(0,72,1)
lw = 3
# Generate figure
fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharey = True)
for i in range(3):
    axs[i].plot(t,fBias[t,i],'-', linewidth = lw)
    axs[i].plot(t,eBias[t,i],'-', linewidth = lw)
    axs[i].plot(t,tBias[t,i],'-', linewidth = lw)
    axs[i].axhline(y=0, color='black', linestyle='--')
    axs[i].set_xticks(np.arange(3,74,6))
    #axs[i].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Titles
axs[0].set_title('Open Water (a$_0$)')
axs[1].set_title('Thin Ice (a$_1$)')
axs[2].set_title('Thick Ice (a$_2$)')

# Add legend to first subplot
axs[0].legend(['Free', 'EnKF', 'SEnT'], loc = 'upper left', fontsize = 24)


# Set y-ticks
axs[0].set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])

# Label the y-axis
fig.supylabel('Bias')

# Xtick labels
axs[0].set_xticklabels([])
axs[1].set_xticklabels([])
axs[2].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 32)

fig.tight_layout()

# Save Figure
plt.savefig('Figures_final/BIAS_FINAL.png', dpi = 360)

#%%
"""
Ensemble Spread
"""

# # Global Variables for plots
# plt.rcParams['font.size'] = 18
# plt.rcParams['font.family'] = 'Times New Roman'



# aw = 2
# freeData = free[2]
# transportData = transport[aw]
# start = 0
# stop = 365

# # Create the figure object
# fig, axs = plt.subplots(3, 2, figsize=(14, 8), sharey = True, sharex = True)
# plt.subplots_adjust(wspace=0.02)
# plt.subplots_adjust(hspace=0.36)

# # Plot open water fraction
# freeData.plot_a0(start,stop, axs[0,0]) 
# transportData.plot_a0(start,stop,axs[0,1]) 

# # Thin ice
# freeData.plotIce(start, stop,0, axs[1,0])
# transportData.plotIce(start, stop,0, axs[1,1])

# # Thick ice
# freeData.plotIce(start, stop,1, axs[2,0])
# transportData.plotIce(start, stop,1, axs[2,1])

# # Set x-ticks
# axs[0,0].set_xticks(np.arange(10,360,30))

# # Xtick labels
# axs[0,0].set_xticklabels([])
# axs[0,1].set_xticklabels([])
# axs[1,1].set_xticklabels([])
# axs[1,0].set_xticklabels([])
# axs[2,0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# axs[2,1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# #axs[2,0].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
# #axs[2,1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])



# # Set yticks
# axs[0, 0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# # Set ylabels
# #axs[1,0].set_ylabel('Fractional Coverage', fontsize = 30)
# fig.supylabel('Fractional Coverage')

# # Set titles
# axs[0,0].set_title('Open Water (a$_0$)')
# axs[0,1].set_title('SEnT: Open Water')
# axs[1,0].set_title('Thin Ice (a$_1$)')
# axs[1,1].set_title('SEnT: Thin Ice')
# axs[2,0].set_title('Thick Ice (a$_2$)')
# axs[2,1].set_title('SEnT: Thick Ice')

# # Legend
# axs[0,0].legend(['Truth', 'Ensemble Mean', 'Ensemble'], fontsize = 16)

# fig.tight_layout()

# # Save Figure
# plt.savefig('Figures_final/Ensemble_spread.png', dpi = 360)

# #%%
# """
# Ensemble Spread No Assimilation
# """

# import dataOutput as da

# # Global Variables for plots
# plt.rcParams['font.size'] = 18
# plt.rcParams['font.family'] = 'Times New Roman'



# aw = 2
# freeData = free[2]
# transportData = transport[aw]
# start = 160
# stop = 300

# # Create the figure object
# fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey = True, sharex = True)
# plt.subplots_adjust(wspace=0.05)
# #plt.subplots_adjust(hspace=0.36)

# # Plot open water fraction
# freeData.plot_a0(start,stop, axs[0]) 
# axs[0].set_title('Open Water Fraction (a$_0$)')

# # Thin ice
# freeData.plotIce(start, stop,0, axs[1])
# axs[1].set_title('Thin Ice (a$_1$)')

# # Figure labels
# axs[0].set_ylabel('Fractional Coverage')
# axs[0].legend(['Truth', 'Ensemble Member'], fontsize = 16)
# #axs[0].axvline(x=263, color='r', linestyle='--', linewidth=2)
# #axs[1].axvline(x=263, color='r', linestyle='--', linewidth=2)


# axs[0].set_xticks(np.arange(start+5,stop+5,30))
# axs[1].set_xticks(np.arange(start+5,stop+5,30))
# axs[0].set_xticklabels(['Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
# axs[1].set_xticklabels(['Jun', 'Jul', 'Aug', 'Sep', 'Oct'])

# # Save Figure
# plt.savefig('Figures_final/Ensemble_spread_no_assimilation.png', dpi = 360)

#%%

"""
Ensemble spread all ice categories just the mean and the refence value

"""

# Global Variables for plots
plt.rcParams['font.size'] = 26
plt.rcParams['font.family'] = 'Times New Roman'

# Dataset
freeData = free[2]

# Plot figure
plt.figure(figsize=(15, 8))
freeData.plotIceCats(0, 365)

# Made figure pretty
plt.ylabel('Fractional Coverage')
plt.title('Daily Changes in Sea Ice Thickness')
plt.legend(['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4','Cat 5'],bbox_to_anchor=(1.25, 1), loc='upper right')
plt.xticks(np.arange(5,360,30), labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()

# Save Figure
plt.savefig('Figures_final/Overview_ice_categories.png', dpi = 360)


#%%

"""
Ensemble evolution on simplex

"""

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fs = 14
f = 'Times New Roman'

def aggregateData(data):
    V1 = 1-np.sum(data, axis = 2)                                       # open water fraction
    V2 = data[:,:,0]                                                    # Cats 1 
    V3 = data[:,:,2] + data[:,:,3] +  data[:,:,4] + data[:,:,1]         # Cat 2, 3, 4
    return np.stack((V1, V2, V3),axis = 2)

def aggregateTruth(truth):
    t1 = 1 - np.sum(truth, axis = 1)
    t2 = truth[:,0]
    t3 = truth[:,1] + truth[:,2]+truth[:,3]+truth[:,4]
    return np.stack((t1,t2,t3), axis = 1)

def plotSimplex(forecast, analysis, truth, ax):

    # Define vertices
    vertices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    # Create surface
    ax.add_collection3d(Poly3DCollection([vertices], facecolors='white', linewidths=1, edgecolors='black', alpha=0.65))
    #ax.grid(False)
    ax.view_init(elev=40, azim=45)

    # Add data to plot
    fore = forecast*1.02
    anal = analysis*1.02
    ax.scatter(fore[:,0], fore[:,1], fore[:,2], s = 50, zorder = 250, alpha = 0.7, edgecolor = 'black', linewidths = 1.5)
    ax.scatter(anal[:,0], anal[:,1], anal[:,2], s = 50, zorder = 250, alpha = 0.7, edgecolor = 'black', linewidths = 1.5)
    ax.scatter(truth[0]*1.03, truth[1]*1.03,truth[2]*1.03, s = 350, marker = '*', color = 'green', edgecolor = 'black', alpha = 1)
    #ax.scatter(anal[:,0], np.ones(len(fore)), np.zeros(len(fore)), color = 'black', alpha = 0.1)
    ax.set_xlabel('Open Water Fraction', fontsize = fs+4, labelpad = 10)
    ax.set_ylabel('Thin Ice', fontsize = fs+4, labelpad = 10)
    ax.set_zlabel('Thick Ice', fontsize = fs+4, labelpad = 10)
    ax.set_xticklabels(labels = [0,0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = fs, fontname = f)
    ax.set_yticklabels(labels = [0,0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = fs, fontname = f)
    ax.set_zticklabels(labels = [0,0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = fs, fontname = f)
   



# Load Data SEnT
fileName = "../2step/output_No10_Ne80_ol0_2024-11-11.h5"
#fileName = "../EnKF/output_No10_Ne80_ol0_2024-10-07_ref6.h5"
#fileName = "../2step_simple/output_No10_Ne80_ol0_2024-11-13.h5"
F = h5py.File(fileName, 'r')  
data = 'aicen'

# Extract data
analysis = F['analysis'][data]               
forecast = F['forecast'][data]
truth = F['ref'][data]
obs = F['observations']
x0 = obs['k'][:]/obs['No'][:]

# Aggregate data
Vf = aggregateData(forecast)
Va = aggregateData(analysis)

# Aggregate truth
T = aggregateTruth(truth)

# Full figure
fig = plt.figure(figsize=(16,10))
plt.subplots_adjust(hspace = 0.25)


# SEnT  
fs2 = 22
# First plot
day1 = 205
ax1 = fig.add_subplot(2,3,1, projection='3d')
plotSimplex(Vf[day1], Va[day1], T[day1], ax1)
ax1.set_title('July 26', fontsize = fs2)

# Second plot
day2 = 215
ax2 = fig.add_subplot(2,3,2, projection='3d')
plotSimplex(Vf[day2], Va[day2], T[day2], ax2)
ax2.set_title('Aug 5', fontsize = fs2)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_zlabel('')

# Third plot
day3 = 235
ax3 = fig.add_subplot(2,3,3, projection='3d')
plotSimplex(Vf[day3], Va[day3], T[day3], ax3)
ax3.set_title('Aug 25', fontsize = fs2)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_zlabel('')


# Load Data Scale
fileName = "../2step_simple/output_No10_Ne80_ol0_2024-11-13.h5"
F = h5py.File(fileName, 'r')  
# Extract data
analysis = F['analysis'][data]               
forecast = F['forecast'][data]
truth = F['ref'][data]
obs = F['observations']
x0 = obs['k'][:]/obs['No'][:]

# Aggregate data
Vf = aggregateData(forecast)
Va = aggregateData(analysis)

# Aggregate truth
T = aggregateTruth(truth)

# Scale
# First plot
ax4 = fig.add_subplot(2,3,4, projection='3d')
plotSimplex(Vf[day1], Va[day1], T[day1], ax4)

# Second plot
ax5 = fig.add_subplot(2,3,5, projection='3d')
plotSimplex(Vf[day2], Va[day2], T[day2], ax5)
ax5.set_xlabel('')
ax5.set_ylabel('')
ax5.set_zlabel('')

# Third plot
ax6 = fig.add_subplot(2,3,6, projection='3d')
plotSimplex(Vf[day3], Va[day3], T[day3], ax6)
ax6.set_xlabel('')
ax6.set_ylabel('')
ax6.set_zlabel('')

plt.savefig('Figures_final/simplex_evolution.png', dpi = 360)


#%%
"""
Toy Model

"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import sys
from pathlib import Path
# Add a line here that looks for simplex_assimilate in a new path
# =============================================================================
sys.path.append('/Users/kabo1917/simplex_assimilate')
# print(sys.path)
from simplex_assimilate import simple_class_transport as sct 

# Example: Red-Yellow-Green colormap
cdict = {'red':   [[0.0, 1.0, 1.0],  # Position, Red start, Red end
                   [0.5, 1.0, 1.0],
                   [1.0, 0.0, 0.0]],
         'green': [[0.0, 0.0, 0.0],
                   [0.5, 1.0, 1.0],
                   [1.0, 1.0, 1.0]],
         'blue':  [[0.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]]}
custom_cmap = LinearSegmentedColormap('CustomMap', cdict)

fs = 13

def plotSimplex(data, colors, title, ax):
    # Define vertices
    vertices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    # Create surface
    ax.add_collection3d(Poly3DCollection([vertices], facecolors='white', linewidths=1, edgecolors='black', alpha=0.8))
    #ax.grid(False)
    ax.view_init(elev=40, azim=45)

    # Add data to plot
    data_plot = data*1.01
    ax.scatter(data_plot[:,0], data_plot[:,1], data_plot[:,2], c = colors,cmap = 'cool', s = 30, zorder = 100, alpha = 1.0)
    ax.scatter(data_plot[:,0], np.ones(len(data)), np.zeros(len(data)), color = 'black', alpha = 1.0)
    ax.set_title(title)
    ax.set_xlabel('Observation Axis', labelpad =  15)
    ax.set_xticks(np.arange(0,1.2, 0.2))
    ax.set_yticks(np.arange(0,1.2, 0.2))
    ax.set_zticks(np.arange(0,1.2, 0.2))
    ax.set_xticklabels(labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = fs, fontname = f)
    ax.set_yticklabels(labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = fs, fontname = f)
    ax.set_zticklabels(labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = fs, fontname = f)
   

# # Global Variables for plots
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'


# Example dataset
data = np.array([[0.4, 0.6, 0],
                [0.35, 0.65, 0],
                [0.3, 0.5, 0.2],
                [0.4,0.5,0.1],
                [0.2, 0.6, 0.2],
                [0.15, 0.75, 0.1],
                [0.1, 0.85, 0.05],
                [0.1, 0.6, 0.3],
                [0.6, 0.2, 0.2],
                [0.7, 0.1, 0.2],
                [0.5, 0.3, 0.2],
                [0, 0.6, 0.4],
                [0,0.7, 0.3],
                [0.7,0,0.3],
                [0.8,0,0.2]])

x0_posterior = np.array([0.21, 0.32, 0.41, 0.33, 0.2, 0.15, 0.13, 0.1, 0.3, 0.24, 0.1, 0, 0, 0.4, 0.1])   # Assuming we got this from step 1



# Find the posterior ensemble using Simplex Assimilate
x0_prior = data[:,0]
x0 = np.array([x0_prior, x0_posterior]).T
rng = np.random.default_rng(seed = 11)
X = sct.transport_pipeline(data, x0, rng)

# Find the posterior ensemble using Scaling
scale_factor = np.divide((1-x0_posterior),(1-x0_prior))
Xscale = np.zeros_like(data)
Xscale[:,0] = x0_posterior 
Xscale[:,1] = np.multiply(scale_factor, data[:,1])
Xscale[:,2] = np.multiply(scale_factor, data[:,2])


# Create the figure
fig = plt.figure(figsize=(14,5))
plt.subplots_adjust(wspace = .28)

# Add subplot for the prior
ax1 = fig.add_subplot(1,3,1, projection='3d')
# Determine classes, and alpha parameters for each dirichlet
classes, class_idxs, alphas, pi = sct.est_mixed_dirichlet(data)
plotSimplex(data, class_idxs, 'Prior', ax1)


# Add subplot for Scale
ax2 = fig.add_subplot(1,3,2, projection='3d')
# Determine classes, and alpha parameters for each dirichlet
classes, class_idxs, alphas, pi = sct.est_mixed_dirichlet(Xscale)
plotSimplex(Xscale, class_idxs, 'Scale', ax2)
ax2.set_xlabel('')

# Add subplot for SEnT
ax3 = fig.add_subplot(1,3,3, projection='3d')
# Determine classes, and alpha parameters for each dirichlet
classes, class_idxs, alphas, pi = sct.est_mixed_dirichlet(X)
plotSimplex(X, class_idxs, 'SEnT', ax3)
ax3.set_xlabel('')
 

plt.savefig('Figures_final/toy model.png', dpi = 360)





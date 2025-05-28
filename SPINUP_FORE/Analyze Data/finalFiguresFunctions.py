#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Figures Functions

@author: kabo1917
"""

import dataOutput as da
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def loadFiles(f20, f40, f60, f80):
    return(f20, f40, f60, f80)

# Takes as an input the file names and the variable of interest (aicen or vicen)
# Outputs a list of dataOutput objects 
def combineDatasets(files, var):
    as0 = da.dataOutput(files[0],var)
    as1 = da.dataOutput(files[1],var)
    as2 = da.dataOutput(files[2],var)
    as3 = da.dataOutput(files[3],var)
    as4 = da.dataOutput(files[4],var)
    return [as0, as1, as2, as3, as4]

def combineAEM(datasets):
    AEM = np.zeros((73*5,6))
    AEM[0:73] = datasets[0].computeAEM()[np.arange(0,365,5)]
    AEM[73:73*2] = datasets[1].computeAEM()[np.arange(1,365,5)]
    AEM[73*2:73*3] = datasets[2].computeAEM()[np.arange(2,365,5)]
    AEM[73*3:73*4] = datasets[3].computeAEM()[np.arange(3,365,5)]
    AEM[73*4:73*5] = datasets[4].computeAEM()[np.arange(4,365,5)]
    return AEM

def combineCRPS(datasets):
    CRPS = np.zeros((73*5,6))
    CRPS[0:73] = datasets[0].computeCRPS(np.arange(0,365,5))
    CRPS[73:73*2] = datasets[1].computeCRPS(np.arange(1,365,5))
    CRPS[73*2:73*3] = datasets[2].computeCRPS(np.arange(2,365,5))
    CRPS[73*3:73*4] = datasets[3].computeCRPS(np.arange(3,365,5))
    CRPS[73*4:73*5] = datasets[4].computeCRPS(np.arange(4,365,5))
    return CRPS

def combineBias(datasets):
    AEM = np.zeros((73*5,6))
    AEM[0:73] = datasets[0].computeBias()[np.arange(0,365,5)]
    AEM[73:73*2] = datasets[1].computeBias()[np.arange(1,365,5)]
    AEM[73*2:73*3] = datasets[2].computeBias()[np.arange(2,365,5)]
    AEM[73*3:73*4] = datasets[3].computeBias()[np.arange(3,365,5)]
    AEM[73*4:73*5] = datasets[4].computeBias()[np.arange(4,365,5)]
    return AEM


def AEMfigure(transport, enkf, scale, free, particle, axs):
    # Combine all the assimilation datasets
    tAEM = combineAEM(transport)
    eAEM = combineAEM(enkf)
    sAEM = combineAEM(scale)
    fAEM = combineAEM(free)
    pAEM = combineAEM(particle)

    # Time Average
    tAEM = tAEM.mean(axis = 0)
    eAEM = eAEM.mean(axis = 0)
    sAEM = sAEM.mean(axis = 0)
    fAEM = fAEM.mean(axis = 0)
    pAEM = pAEM.mean(axis = 0)

    mas = 11
    lw = 2
    # Plot
    axs.plot(fAEM[0:], '^--', ms = mas, linewidth = lw)
    axs.plot(eAEM[0:],'o--', ms = mas, linewidth = lw)
    axs.plot(tAEM[0:],'s--', ms = mas, linewidth = lw)
    axs.plot(sAEM[0:],'*--', ms = mas+4, linewidth = lw)
    axs.plot(pAEM[0:],'D--', ms = mas, linewidth = lw)
    axs.set_xticks(np.arange(6))
    axs.set_xticklabels(('a$_0$', 'a$_1$','a$_2$','a$_3$','a$_4$','a$_5$'), fonttype = 'Times New Roman')

def CRPSfigure(transport, enkf, scale, free, particle, axs):
    # Combine all assimilation datasets
    tCRPS = combineCRPS(transport)
    eCRPS = combineCRPS(enkf)
    sCRPS = combineCRPS(scale)
    fCRPS = combineCRPS(free)
    pCRPS = combineCRPS(particle)

    # Time averaged mean
    tCRPS = tCRPS.mean(axis = 0)
    eCRPS = eCRPS.mean(axis = 0)
    sCRPS = sCRPS.mean(axis = 0)
    fCRPS = fCRPS.mean(axis = 0)
    pCRPS = pCRPS.mean(axis = 0)

    mas = 11
    lw = 2
    # Plot
    axs.plot(fCRPS[0:],'^--',ms = mas, linewidth = lw)
    axs.plot(eCRPS[0:],'o--',ms = mas, linewidth = lw)
    axs.plot(tCRPS[0:],'s--',ms = mas, linewidth = lw)
    axs.plot(sCRPS[0:],'*--',ms = mas, linewidth = lw)
    axs.plot(pCRPS[0:],'D--',ms = mas, linewidth = lw)
    axs.set_xticklabels(('a$_0$', 'a$_1$','a$_2$','a$_3$','a$_4$','a$_5$'))

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
    ax.set_xlabel('Open Water Fraction', fontsize = 16)
    ax.set_ylabel('Thin Ice', fontsize = 16)
    ax.set_zlabel('Thick Ice', fontsize = 16)
    ax.set_xticklabels(labels = [0,0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 16)
    ax.set_yticklabels(labels = [0,0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 16)
    ax.set_zticklabels(labels = [0,0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 16)
   
    
    
    
    
    
    
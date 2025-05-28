#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:26:12 2024

@author: kabo1917
"""

"""
Class dataOutput takes in a filename for an h5 file and a variable of interest and then defines the attributes to be datasets of that variable

Methods:



"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import CRPS.CRPS as CRPS

def aggregateData(data,day):
     data = data[day,:,:]
     V1 = 1-np.sum(data, axis = 1)                                     # open water fraction
     V2 = data[:,0]                                                    # Cats 1 
     V3 = data[:,1] + data[:,2] + data[:,3] + data[:,4]                # Cat 2, 3, 4
     return np.stack((V1, V2, V3),axis = 1)
 
def aggregateTruth(truth, day):
     truth = truth[day]
     t1 = 1 - np.sum(truth)
     t2 = truth[0]
     t3 = truth[1] + truth[2]+truth[3]+truth[4]
     return np.stack((t1,t2,t3))

class dataOutput(object):
    # Initialize class and set different data to be the attributes
    def __init__(self, filename, var): 
        data = h5py.File(filename, 'r') 
        self.analysis = data['analysis'][var]
        self.forecast = data['forecast'][var]
        self.obs = [data['observations']['No'], data['observations']['k']]
        self.ref = data['ref'][var]
        self.SIC = np.sum(self.analysis, axis = 2)
        self.SICf = np.sum(self.forecast, axis = 2)
    
    def __add__(self, other1):
        return dataOutput(self + other1)
    
    # Plot SIC for a given range of days for a single output file
    def plotSIC(self, st, et, label):
        x = range(st,et)
        plt.plot(x, self.SIC[st:et], c = 'grey', linewidth = 0.3)
        plt.plot(x, np.sum(self.ref[st:et], axis = 1), c = 'g')
        plt.ylabel("SIC", fontsize = 10)
        plt.ylim([0,1.02]) 
        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 10)
        plt.title(label, fontsize = 12)
        
    # Plot SIC for a given range of days for a single output file
    def plotSICall(self, enkf, transport, scale, st, et, savelabel):
        x = range(st,et)
        fig, axs = plt.subplots(2,2, sharex = True, sharey = True)
        axs[0,0].plot(x, self.SIC[st:et], c = 'grey', linewidth = 0.3)
        axs[0,0].plot(x, np.sum(self.ref[st:et], axis = 1), c = 'g')
        axs[0,0].set_title("Free", fontsize = 10)
        
        axs[0,1].plot(x, enkf.SIC[st:et], c = 'grey', linewidth = 0.3)
        axs[0,1].plot(x, np.sum(self.ref[st:et], axis = 1), c = 'g')
        axs[0,1].set_title("EnKF", fontsize = 10)
        
        axs[1,0].plot(x, transport.SIC[st:et], c = 'grey', linewidth = 0.3)
        axs[1,0].plot(x, np.sum(self.ref[st:et], axis = 1), c = 'g')
        axs[1,0].set_title("Transport", fontsize = 10)
        
        axs[1,1].plot(x, scale.SIC[st:et], c = 'grey', linewidth = 0.3)
        axs[1,1].plot(x, np.sum(self.ref[st:et], axis = 1), c = 'g')
        axs[1,1].set_title("Scale", fontsize = 10)
        
        fig.suptitle('SIC')
        fig.savefig(savelabel, dpi = 360)
    
    # Plot open water fraction
    # def plot_a0(self, st, et, label):
    #     x = range(st,et)
    #     plt.plot(x, 1-np.sum(self.ref[st:et], axis = 1), c = 'lime')
    #     plt.plot(x, 1-np.mean(self.SIC,1)[st:et], c = 'black')
    #     plt.plot(x, 1-self.SIC[st:et], c = 'grey', linewidth = 0.3)
    #     plt.plot(x, 1-np.sum(self.ref[st:et], axis = 1), c = 'lime')
    #     plt.plot(x, 1-np.mean(self.SIC,1)[st:et], c = 'black')
    #     #plt.ylabel("Fractional Coverage", fontsize = 13)
    #     plt.ylim([0,1.02]) 
    #     plt.xticks(np.arange(0,360,30),['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 14, fontname = "Times New Roman")
    #     plt.yticks(fontsize = 12, fontname = "Times New Roman")
    #     plt.title(label, fontsize = 18, fontname = "Times New Roman")
        
    def plot_a0(self, st, et, axs):
        x = range(st,et)
        axs.plot(x, 1-np.sum(self.ref[st:et], axis = 1), c = 'cyan')
        #axs.plot(x, 1-np.mean(self.SIC,1)[st:et], c = 'black')
        axs.plot(x, 1-self.SIC[st:et], c = 'grey', linewidth = 0.7)
        axs.plot(x, 1-np.sum(self.ref[st:et], axis = 1), c = 'cyan', lw = 4)
        #axs.plot(x, 1-np.mean(self.SIC,1)[st:et], c = 'black', lw = 3)
        axs.set_ylim([0,1.02]) 
        
   # Plot SIC histogram for a single day for a single output file
    def sicHist(self, date, runType):
        plt.hist(self.SIC[date])
        plt.title(runType + ': SIC Histogram Day ' + str(date))
        plt.xlim([0,1])
    
    # Plot an ice category for a given range of days for a single output file
    # def plotIce(self, st, et, cat, label):
    #     x = range(st,et)
    #     plt.plot(x, self.ref[st:et,cat], c = 'lime')
    #     plt.plot(x,np.mean(self.analysis,1)[st:et,cat], c = 'black')
    #     plt.plot(x, self.analysis[st:et,:,cat], c = 'grey', linewidth = 0.3)
    #     plt.plot(x, self.ref[st:et,cat], c = 'lime')
    #     plt.plot(x,np.mean(self.analysis,1)[st:et,cat], c = 'black')
    #     plt.ylim([0,1])
    #     plt.xticks(np.arange(0,360,30),['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 14, fontname = "Times New Roman")
    #     plt.yticks(fontsize = 12, fontname = "Times New Roman")
    #     plt.ylim([0,1.02]) 
    #     plt.title(label, fontsize = 18, fontname = "Times New Roman")
        
    def plotIce(self, st, et, cat, axs):
        x = range(st,et)
        axs.plot(x, self.ref[st:et,cat], c = 'cyan')
        axs.plot(x,np.mean(self.analysis,1)[st:et,cat], c = 'black')
        axs.plot(x, self.analysis[st:et,:,cat], c = 'grey', linewidth = 0.7)
        axs.plot(x, self.ref[st:et,cat], c = 'cyan', lw = 4)
        axs.plot(x,np.mean(self.analysis,1)[st:et,cat], c = 'black', lw = 3)
        
    def plotIceCats(self, st, et):
        x = range(st,et)
        w = 3
        # Plot the ensemble mean
        plt.plot(x,np.mean(self.analysis,1)[st:et,0],'-', c = 'dodgerblue', lw = w)
        plt.plot(x,np.mean(self.analysis,1)[st:et,1],'-', c = 'teal', lw = w)
        plt.plot(x,np.mean(self.analysis,1)[st:et,2],'-', c = 'blue', lw = w)
        plt.plot(x,np.mean(self.analysis,1)[st:et,3],'-', c = 'plum', lw = w)
        plt.plot(x,np.mean(self.analysis,1)[st:et,4],'-', c = 'indigo', lw = w)
        # Plot the reference value with a dashed line
        plt.plot(x, self.ref[st:et,0],'--', c = 'dodgerblue', lw = w)
        plt.plot(x, self.ref[st:et,1],'--', c = 'teal', lw = w)
        plt.plot(x, self.ref[st:et,2],'--', c = 'blue', lw = w)
        plt.plot(x, self.ref[st:et,3],'--', c = 'plum', lw = w)
        plt.plot(x, self.ref[st:et,4],'--', c = 'indigo', lw = w)
        
        

    # Plot an ice category for a given range of days for a single output file
    def plotIceall(self, EnKF, transport, scale, st, et, cat, label):
        x = range(st,et)
        fig, axs = plt.subplots(2,2, sharex = True, sharey = True)
        axs[0,0].plot(x, self.analysis[st:et,:,cat], c = 'grey', linewidth = 0.3)
        axs[0,0].plot(x, self.ref[st:et,cat], c = 'cyan')
        axs[0,0].set_title("Free", fontsize = 10)
        
        axs[0,1].plot(x, EnKF.analysis[st:et,:,cat], c = 'grey', linewidth = 0.3)
        axs[0,1].plot(x, EnKF.ref[st:et,cat], c = 'cyan')
        axs[0,1].set_title("EnKF", fontsize = 10)
        
        axs[1,0].plot(x, transport.analysis[st:et,:,cat], c = 'grey', linewidth = 0.3)
        axs[1,0].plot(x, transport.ref[st:et,cat], c = 'g')
        axs[1,0].set_title("Transport", fontsize = 10)
        
        axs[1,1].plot(x, scale.analysis[st:et,:,cat], c = 'grey', linewidth = 0.3)
        axs[1,1].plot(x, scale.ref[st:et,cat], c = 'g')
        axs[1,1].set_title("Scale", fontsize = 10)
        
        fig.suptitle(label)
        fig.savefig('Figures/' + label + '.png', dpi = 360)
        
        
    # Plot ICE histogram for a single day for a single output file
    def iceHist(self, date, cat, runType):
        plt.hist(self.analysis[date,:,cat])
        plt.title(runType + ': Ice Cat ' + str(cat) +' Histogram Day ' + str(date))
        plt.xlim([0,1])
        
    # Compare across ice categories for a range of days for a single output file
    def compareIce(self, st, et, date, title):
        x = range(st,et)
        fig, axs = plt.subplots(2,3)
        for i in range(3):
            axs[0, i].plot(x, self.analysis[st:et,:,i], c = 'grey', linewidth = 0.3)
            axs[0, i].plot(x, self.ref[st:et,i], c = 'g')
            axs[0, i].axvline(date)
            axs[0, i].set_ylim([0, 1])
            axs[0, i].set_title('Ice Cat: '+ str(i))
        for i in range(3):
            axs[1, i].hist(self.analysis[date,:,i])
            axs[1, i].set_xlim([0, 1])
            axs[1, i].set_ylim([0, 50])
        fig.suptitle(title)
        for axs in fig.get_axes():
            axs.label_outer()
            
    def compareOutputIce(self, other, other2, st, et, cat, title):
        x = range(st,et)
        fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
        axs[0].plot(x, self.analysis[st:et, :, cat], c = 'grey', linewidth = 0.3)
        axs[0].plot(x, self.ref[st:et, cat], c = 'g')
        axs[0].set_title(title[0])
        axs[1].plot(x, other.analysis[st:et, :, cat], c = 'grey', linewidth = 0.3)
        axs[1].plot(x, other.ref[st:et, cat], c = 'g')
        axs[1].set_title(title[1])
        axs[2].plot(x, other2.analysis[st:et, :, cat], c = 'grey', linewidth = 0.3)
        axs[2].plot(x, other2.ref[st:et, cat], c = 'g')
        axs[2].set_title(title[2])
    
    # Only use with the vsnon variable 
    def compareSnow(self, other, other2, st, et, title):
        x = range(st,et)
        totalSnow = np.sum(self.analysis, axis = 2)
        totalSnow_other = np.sum(other.analysis, axis = 2)
        totalSnow_other2 = np.sum(other.analysis, axis = 2)
        fig, axs = plt.subplots(1,3, sharex = True, sharey = True)
        axs[0].plot(x, totalSnow[st:et, :], c = 'grey', linewidth = 0.3)
        axs[0].plot(x, np.sum(self.ref[st:et], axis = 1), c = 'g')
        axs[0].set_title(title[0])
        axs[1].plot(x, totalSnow_other[st:et, :], c = 'grey', linewidth = 0.3)
        axs[1].plot(x, np.sum(other.ref[st:et], axis = 1), c = 'g')
        axs[1].set_title(title[1])
        axs[2].plot(x, totalSnow_other2[st:et, :], c = 'grey', linewidth = 0.3)
        axs[2].plot(x, np.sum(other2.ref[st:et], axis = 1), c = 'g')
        axs[2].set_title(title[2])
        
    def forecastVanalysis(self, cat, date):
        plt.hist(self.forecast[date,:,cat], edgecolor ='blue', alpha = 0.4, range = (0,1))
        plt.hist(self.analysis[date,:,cat], edgecolor ='orange', alpha = 0.3, range = (0,1) )
        plt.scatter(self.ref[date,cat], 0.5, s = 100, marker = '*' , color = 'g')
        plt.xlim([0,1])
        plt.legend(['Reference','Forecast', 'Analysis'])
        
    def computeRMSE(self):
        # Computes RMSE everyday, includes open water fraction
        analysis = self.analysis
        truth = self.ref
        size = np.shape(analysis)
        MSE = np.zeros((size[0],size[2]+1))
        N = np.shape(analysis)[1]
        for i in range(N):
            MSE[:,0] += ((1-np.sum(self.ref, axis = 1))-(1-self.SIC[:,i]))**2
        for i in range(N):
            MSE[:,1:] += (analysis[:,i,:]-truth)**2
        MSE = np.sqrt(1/N*MSE)
        return MSE
    
    def computeAEM(self):
        # Computes AEM everyday, includes open water fraction
        ensMean = np.mean(self.analysis, axis = 1)
        a0Mean = 1-np.mean(self.SIC, axis = 1)
        size = np.shape(ensMean)
        AEM = np.zeros((size[0],size[1]+1))
        AEM[:,0] = abs((1-np.sum(self.ref, axis = 1))-a0Mean)
        AEM[:,1:] = abs(self.ref-ensMean)
        return AEM
    
    def computeBias(self):
        # Computes bias everyday, includes open water fraction
        ensMean = np.mean(self.analysis, axis = 1)
        a0Mean = 1-np.mean(self.SIC, axis = 1)
        size = np.shape(ensMean)
        Bias = np.zeros((size[0],size[1]+1))
        Bias[:,0] = (a0Mean -(1-np.sum(self.ref, axis = 1)))
        Bias[:,1:] = (ensMean -self.ref)
        return Bias
    
    def computeCRPS(self, t):
        # Computes CRPS every assimilation cycle given by t, includes open water fraction
        shape = np.shape(self.analysis)
        cats = shape[2]+1
        crps = np.zeros((np.size(t),cats))
        
        for i in range(np.size(t)):
            crps[i,0] = CRPS(1-self.SIC[t[i],:], (1-np.sum(self.ref, axis = 1))[t[i]]).compute()[0]
        for j in range(5):
            for i in range(np.size(t)):
                crps[i,j+1] = CRPS(self.analysis[t[i],:,j],self.ref[t[i],j]).compute()[0]
        return crps


    def plotSimplex(self, ax, day):
        forecast = aggregateData(self.forecast, day)
        analysis = aggregateData(self.analysis, day)
        truth = aggregateTruth(self.ref, day)
        
        # Define vertices
        vertices = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    
        # Create surface
        ax.add_collection3d(Poly3DCollection([vertices], facecolors='white', linewidths=1, edgecolors='black', alpha=0.65))
        #ax.grid(False)
        ax.view_init(elev=40, azim=45)
    
        # Add data to plot
        fore = forecast*1.02
        anal = analysis*1.02
        ax.scatter(fore[:,0], fore[:,1], fore[:,2], s = 50, zorder = 250, alpha = 0.6, edgecolor = 'black', linewidths = 1.5)
        ax.scatter(anal[:,0]*1.01, anal[:,1]*1.01, anal[:,2]*1.01, s = 50, zorder = 250, alpha = 0.7, edgecolor = 'black', linewidths = 1.5)
        ax.scatter(truth[0]*1.1, truth[1]*1.1,truth[2]*1.1, s = 400, marker = '*', color = 'chartreuse', edgecolor = 'black', alpha = 1)
        #ax.scatter(anal[:,0], np.ones(len(fore)), np.zeros(len(fore)), color = 'black', alpha = 0.1)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='z', labelsize=12)
        ax.set_xlabel('Open Water Fraction', fontsize = 13)    
   
            
# Write functions separate from class 
def standardize(array):
     dailyMean = array.mean(axis = 1)
     dailySTD = np.std(array, axis = 1)
     standard = np.zeros(np.shape(array))
     for i in range(np.shape(array)[0]):
         standard[i,:] = (array[i,:]-dailyMean[i])/dailySTD[i]
     return standard
 


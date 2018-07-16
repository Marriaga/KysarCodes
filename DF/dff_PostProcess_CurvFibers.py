#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:48:26 2018

@author: df
"""

import os, os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pandas as pd

from sklearn import datasets, linear_model

from statsmodels.formula.api import ols
import seaborn

#myPath = '/Users/df/Documents/myGits/FibersResultsXLSX/'
#csv_path = myPath + 'Curvature_and_Directionality_Results_T.csv'
#Top_file = myPath + 'Curvature_and_Directionality_Results_T.xlsx'


#%% to plot the 1VMU model: 
def plot_PDF_1VM( locs, kaps, wvm, wu, Position=None ):
    scal_ = 0.5
    fig, ax = plt.subplots(1, 1, figsize=(9,4))
    titl = 'PDF of Mixture Model 1VMU - ' + Position
    ax.set_title(titl)
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    x_ = np.linspace( lim_l, lim_u, 100 )
    x_ax = np.degrees(x_)
    jj = 0
    for mu, kap, pvm, pu in zip(locs, kaps, wvm, wu):
        jj += 1
        f_VM = stats.vonmises( kap, mu, scal_ )
        f_Un = stats.uniform( loc=lim_l, scale=lim_u-lim_l )
        X_temp = pvm*f_VM.pdf ( x_ ) + pu*f_Un.pdf( x_ )
        ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-')
        #ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-', \
         #       label='DP {} '.format(jj))
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$f(\theta)$', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
    #ax.legend(loc=9)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), \
     #         fancybox=True, shadow=True, ncol=9)

    return fig, ax

# to plot the 2VMU model: 
def plot_PDF_2VM( locs1, kaps1, wvm1, locs2, kaps2, wvm2, wu, Position=None ):
    scal_ = 0.5
    fig, ax = plt.subplots(1, 1, figsize=(9,4))
    titl = 'PDF of Mixture Model 2VMU - ' + Position
    ax.set_title(titl)
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    x_ = np.linspace( lim_l, lim_u, 100 )
    x_ax = np.degrees(x_)
    jj = 0
    for mu1, kap1, pvm1, mu2, kap2, pvm2, pu in \
                            zip(locs1, kaps1, wvm1, locs2, kaps2, wvm2, wu):
        jj += 1
        f_VM1 = stats.vonmises( kap1, mu1, scal_ )
        f_VM2 = stats.vonmises( kap2, mu2, scal_ )
        f_Un = stats.uniform( loc=lim_l, scale=lim_u-lim_l )
        X_temp = pvm1*f_VM1.pdf ( x_ ) + pvm2*f_VM2.pdf ( x_ ) + \
                                                            pu*f_Un.pdf( x_ )
        ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-')
        #ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-', \
         #       label='DP {} '.format(jj))
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$f(\theta)$', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
    #ax.legend(loc=9)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), \
     #         fancybox=True, shadow=True, ncol=9)

    return fig, ax

#%% to plot the 1VMU model: 
def plot_PDF_1VM_b( FibersT ):
    
    Position = FibersT.Position[0]
    
    locs = np.radians(FibersT.VM1_Ang)
    kaps = FibersT.VM1_Conc
    wvm = FibersT.VM1_Weig
    wu = FibersT.VM1_Weigu
    
    Angle_KMax = FibersT.Angle_KMax
    Angle_KMin = FibersT.Angle_KMin
    Angle_KMinMag1 = FibersT.Angle_KMinMag1
    Angle_KMinMag2 = FibersT.Angle_KMinMag2
    yones = np.ones(len(Angle_KMax))
    
    scal_ = 0.5
    fig, ax = plt.subplots(1, 1, figsize=(9,4))
    titl = 'PDF of Mixture Model 1VMU - ' + Position
    ax.set_title(titl)
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    x_ = np.linspace( lim_l, lim_u, 100 )
    x_ax = np.degrees(x_)
    jj = 0
    for mu, kap, pvm, pu in zip(locs, kaps, wvm, wu):
        jj += 1
        f_VM = stats.vonmises( kap, mu, scal_ )
        f_Un = stats.uniform( loc=lim_l, scale=lim_u-lim_l )
        X_temp = pvm*f_VM.pdf ( x_ ) + pu*f_Un.pdf( x_ )
        ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-')
        #ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-', \
         #       label='DP {} '.format(jj))
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$f(\theta)$', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)

    ax.plot(Angle_KMax, yones, 'x', color='black', label='KMax' )
    ax.plot(Angle_KMin, 0.9*yones, '+', color='gray', label='KMin' )
    ax.plot(Angle_KMinMag1, 0.8*yones, '^', markerfacecolor='white', markeredgecolor='b', label='MM1' )
    ax.plot(Angle_KMinMag2, 0.7*yones, 'v', markerfacecolor='white', markeredgecolor='g', label='MM2' )
    ax.plot(FibersT.VM1_Ang, wvm, 'o', color='red', label='1VM' )
    
    #ax.legend(loc=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), \
              fancybox=True, shadow=True, ncol=6)

    return fig, ax

#%% to plot the 2VMU model: 
def plot_PDF_2VM_b( FibersT ):
    
    Position = FibersT.Position[0]
    
    locs1 = np.radians(FibersT.VM2_Ang1)
    kaps1 = FibersT.VM2_Conc1
    wvm1 = FibersT.VM2_Weig1
    locs2 = np.radians(FibersT.VM2_Ang2)
    kaps2 = FibersT.VM2_Conc2
    wvm2 = FibersT.VM2_Weig2
    wu = FibersT.VM2_Weigu
    
    Angle_KMax = FibersT.Angle_KMax
    Angle_KMin = FibersT.Angle_KMin
    Angle_KMinMag1 = FibersT.Angle_KMinMag1
    Angle_KMinMag2 = FibersT.Angle_KMinMag2
    yones = np.ones(len(Angle_KMax))
    
    scal_ = 0.5
    fig, ax = plt.subplots(1, 1, figsize=(9,4))
    titl = 'PDF of Mixture Model 2VMU - ' + Position
    ax.set_title(titl)
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    x_ = np.linspace( lim_l, lim_u, 100 )
    x_ax = np.degrees(x_)
    jj = 0
    for mu1, kap1, pvm1, mu2, kap2, pvm2, pu in \
                            zip(locs1, kaps1, wvm1, locs2, kaps2, wvm2, wu):
        jj += 1
        f_VM1 = stats.vonmises( kap1, mu1, scal_ )
        f_VM2 = stats.vonmises( kap2, mu2, scal_ )
        f_Un = stats.uniform( loc=lim_l, scale=lim_u-lim_l )
        X_temp = pvm1*f_VM1.pdf ( x_ ) + pvm2*f_VM2.pdf ( x_ ) + \
                                                            pu*f_Un.pdf( x_ )
        ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-')
        #ax.plot(x_ax, X_temp, color='gray', linewidth=2, linestyle='-', \
         #       label='DP {} '.format(jj))
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$f(\theta)$', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
    
    ax.plot(Angle_KMax, yones, 'x', color='black', label='KMax' )
    ax.plot(Angle_KMin, 0.9*yones, '+', color='gray', label='KMin' )
    ax.plot(Angle_KMinMag1, 0.8*yones, '^', markerfacecolor='white', markeredgecolor='b', label='MM1' )
    ax.plot(Angle_KMinMag2, 0.7*yones, 'v', markerfacecolor='white', markeredgecolor='g', label='MM2' )
    ax.plot(FibersT.VM2_Ang1, FibersT.VM2_Weig1, 's', color='red', label='2VM-1' )
    ax.plot(FibersT.VM2_Ang2, FibersT.VM2_Weig2, 'o', color='m', label='2VM-2' )
    
    #ax.legend(loc=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), \
              fancybox=True, shadow=True, ncol=6)

    return fig, ax

def plot_alpha_disp( FibersTable, Position=None ):
        
    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    titl = 'concentration vs Alpha for modle 1VMU'
    ax.set_title(titl)
    ax.set_xlabel('min(alpha,90-alpha)', fontsize=12)
    ax.set_ylabel('concentration', fontsize=12)
    for pos in Position:
        beta = FibersTable.beta[FibersTable.Position==pos]
        concentration = FibersTable.VM1_Conc[FibersTable.Position==pos]
        #ax.plot(beta, concentration, 'o', label=pos )
        sizes = 100*np.array(FibersTable.VM1_Weig[FibersTable.Position==pos])
        ax.scatter(beta, concentration, s=sizes, alpha=0.5, label=pos )
    #ax.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), \
              fancybox=True, shadow=True, ncol=6)
    
    fig2, ax2 = plt.subplots(2, 3, figsize=(9,6))
    #titl = 'concentration vs Alpha for modle 1VMU'
    #ax.set_title(titl)
    #ax.set_xlabel('max(alpha,90-alpha)', fontsize=12)
    #ax.set_ylabel('concentration', fontsize=12)
    ii = [0, 0, 0, 1, 1]
    jj = [0, 1, 2, 0, 1]
    k1 = 0
    for pos in Position:
        beta = FibersTable.beta[FibersTable.Position==pos]
        concentration = FibersTable.VM1_Conc[FibersTable.Position==pos]
        #ax.plot(beta, concentration, 'o', label=pos )
        sizes = 100*np.array(FibersTable.VM1_Weig[FibersTable.Position==pos])
        ax2[ii[k1],jj[k1]].scatter(beta, concentration, s=sizes, alpha=0.5, label=pos )
        
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        X = np.asarray(beta).reshape(len(beta),1)
        Y = np.asarray(concentration).reshape(len(concentration),1)
        regr.fit(X, Y)
        # Plot outputs
        ax2[ii[k1],jj[k1]].plot(X, regr.predict(X), linewidth=2 )
                
        ax2[ii[k1],jj[k1]].legend()
        
        k1 += 1
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), \
     #         fancybox=True, shadow=True, ncol=6)

#%%
def plot_curvratio_disp( FibersTable, Position=None,xtype="ratio",ytype="concentration",th=100):
        
    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    titl = ytype + ' vs ' + xtype + ' for modle 1VMU'
    ax.set_title(titl)
    maxx=0
    maxy=0
    for pos in Position:
        kmax = abs(FibersTable.KMax[FibersTable.Position==pos])
        kmin = abs(FibersTable.KMin[FibersTable.Position==pos])
        curvr=[]
        for mmax,mmin in zip(kmax,kmin):
            curvr.append(min(mmax,mmin)/max(mmax,mmin))
        beta = FibersTable.beta[FibersTable.Position==pos]
        
        if xtype=="ratio": #use ratio
            XX=curvr
        elif xtype=="beta":
            XX=beta
        elif xtype=="angratio":
            XX=np.arctan(np.sqrt(curvr))*180/3.1415
        
        
        concentration = FibersTable.VM1_Conc[FibersTable.Position==pos]
        
        if ytype=="concentration": #Use Concentration
            YY=concentration
        elif ytype=="dispersion":
            YY=1/concentration
        elif ytype=="normconc":
            YY=concentration/np.array(FibersTable.VM1_Weig[FibersTable.Position==pos])
        
        #ax.plot(beta, dispersion, 'o', label=pos )
        sizes = 100*np.array(FibersTable.VM1_Weig[FibersTable.Position==pos])
        sizes[sizes>th]=sizes[sizes>th]*0.0

        ax.scatter(XX, YY, s=sizes, alpha=0.5, label=pos )
        maxx=max(maxx,max(XX))
        maxy=max(maxy,max(YY[sizes>0]))
        
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        X = np.asarray(XX).reshape(len(XX),1)
        Y = np.asarray(YY).reshape(len(YY),1)
        regr.fit(X, Y)
        # Plot outputs
        ax.plot(X, regr.predict(X), linewidth=2 )
                
    ax.set_xlabel(xtype, fontsize=12)
    ax.set_ylabel(ytype, fontsize=12)
    ax.set_xlim([-0.02*maxx,1.05*maxx])
    ax.set_ylim([-0.02*maxy,1.05*maxy])
    #ax.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), \
              fancybox=True, shadow=True, ncol=6)
        
    return fig, ax


#%%
def plot_angles_1VMU( FibersTable, Position=None ):
    
    pos = Position
    locs = FibersTable.VM1_Ang[FibersTable.Position==pos]
    Angle_KMax = FibersTable.Angle_KMax[FibersTable.Position==pos]
    Angle_KMin = FibersTable.Angle_KMin[FibersTable.Position==pos]
    Angle_KMinMag1 = FibersTable.Angle_KMinMag1[FibersTable.Position==pos]
    Angle_KMinMag2 = FibersTable.Angle_KMinMag2[FibersTable.Position==pos]
    
    fig, ax = plt.subplots(1, 1, figsize=(9,4))
    titl = 'Model location vs Curvature angles - ' + Position
    ax.set_title(titl)
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    x_ax = np.degrees(np.linspace( lim_l, lim_u, 100 ))
    ax.plot(x_ax, x_ax, color='gray', linewidth=2, linestyle='--')
    ax.set_xlim(np.degrees([lim_l, lim_u]))
    ax.set_ylim(np.degrees([lim_l, lim_u]))
    ax.set_xlabel(r'Model location, $\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'Curvature angles, $\theta$ (degrees)', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)

    ax.plot(locs, Angle_KMax, 'o', markerfacecolor='white', markeredgecolor='b', label='KMax' )
    ax.plot(locs, Angle_KMin, 's', markerfacecolor='white', markeredgecolor='g', label='KMin' )
    ax.plot(locs, Angle_KMinMag1, '^', markerfacecolor='white', markeredgecolor='r', label='MM1' )
    ax.plot(locs, Angle_KMinMag2, 'v', markerfacecolor='white', markeredgecolor='m', label='MM2' )
    
    #ax.legend(loc=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), \
              fancybox=True, shadow=True, ncol=6)
    

def scatter_angles_1VMU( FibersTable, Position=None ):
    
    pos = Position
    locs = FibersTable.VM1_Ang[FibersTable.Position==pos]
    Angle_KMax = FibersTable.Angle_KMax[FibersTable.Position==pos]
    Angle_KMin = FibersTable.Angle_KMin[FibersTable.Position==pos]
    Angle_KMinMag1 = FibersTable.Angle_KMinMag1[FibersTable.Position==pos]
    Angle_KMinMag2 = FibersTable.Angle_KMinMag2[FibersTable.Position==pos]
    
    sizes = 10*np.array(FibersTable.VM1_Conc[FibersTable.Position==pos])
    
    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    titl = 'Model location vs Curvature angles - ' + Position
    ax.set_title(titl)
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    x_ax = np.degrees(np.linspace( lim_l, lim_u, 100 ))
    ax.plot(x_ax, x_ax, color='gray', linewidth=2, linestyle='--')
    ax.set_xlim(np.degrees([lim_l, lim_u]))
    ax.set_ylim(np.degrees([lim_l, lim_u]))
    ax.set_xlabel(r'Model location, $\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'Curvature angles, $\theta$ (degrees)', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)

    ax.scatter(locs, Angle_KMax, c='b', s=sizes, alpha=0.5, label='KMax' )
    ax.scatter(locs, Angle_KMin, c='g', s=sizes, alpha=0.5, label='KMin' )
    ax.scatter(locs, Angle_KMinMag1, c='r', s=sizes, alpha=0.5, label='MM1' )
    ax.scatter(locs, Angle_KMinMag2, c='m', s=sizes, alpha=0.5, label='MM2' )
        
    #ax.legend(loc=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), \
              fancybox=True, shadow=True, ncol=6)
    
    
def scatter_angles_RWM( FibersTable, RWM=None ):
    
    #rwm = FibersTable.Name[FibersTable.Name==RWM][0]
    rwm = RWM
    print(rwm)
    locs = FibersTable.VM1_Ang[FibersTable.Name==rwm]
    Angle_KMax = FibersTable.Angle_KMax[FibersTable.Name==rwm]
    Angle_KMin = FibersTable.Angle_KMin[FibersTable.Name==rwm]
    Angle_KMinMag1 = FibersTable.Angle_KMinMag1[FibersTable.Name==rwm]
    Angle_KMinMag2 = FibersTable.Angle_KMinMag2[FibersTable.Name==rwm]
    
    summ = sum(np.array(FibersTable.VM1_Conc[FibersTable.Name==rwm]))
    sizes = 300*np.array(FibersTable.VM1_Conc[FibersTable.Name==rwm])/summ
    
    fig, ax = plt.subplots(1, 1, figsize=(9,6))
    titl = 'Model location vs Curvature angles - RWM: ' + rwm
    ax.set_title(titl)
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    x_ax = np.degrees(np.linspace( lim_l, lim_u, 100 ))
    ax.plot(x_ax, x_ax, color='gray', linewidth=2, linestyle='--')
    ax.set_xlim(np.degrees([lim_l, lim_u]))
    ax.set_ylim(np.degrees([lim_l, lim_u]))
    ax.set_xlabel(r'Fiber principal orientation, $\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'Curvature orientations, $\theta$ (degrees)', fontsize=12)
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)

    ax.scatter(locs, Angle_KMax, c='b', marker="o", s=sizes, alpha=0.5, label='KMax' )
    ax.scatter(locs, Angle_KMin, c='g', marker="s", s=sizes, alpha=0.5, label='KMin' )
    ax.scatter(locs, Angle_KMinMag1, c='r', marker="v", s=sizes, alpha=0.5, label='MM1' )
    ax.scatter(locs, Angle_KMinMag2, c='m', marker="^", s=sizes, alpha=0.5, label='MM2' )
        
    #ax.legend(loc=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), \
              fancybox=True, shadow=True, ncol=6)
    
    angNam = ['KMax', 'KMin', 'MM1', 'MM2']
    marks = ['o','s','v','^','d']
    colrs = ['b', 'g', 'r', 'm']
#    fig2, ax2 = plt.subplots(1, 1, figsize=(9,6))
#    ax2.plot(x_ax, x_ax, color='gray', linewidth=2, linestyle='--')
    membrane = FibersTable['Name'] == RWM
    Position = ['Top','Bottom','Right','Left','Center','Bottom_Left','Bottom_Right']
    for pos, m in zip(Position, marks):
        print('Position: ',pos)
        poss = FibersTable.Position==pos
        aloc = np.array(FibersTable.VM1_Ang[membrane & poss])
        if len(aloc):
            aKMax = np.array(FibersTable.Angle_KMax[membrane & poss])
            aKMin = np.array(FibersTable.Angle_KMin[membrane & poss])
            aKMinMag1 = np.array(FibersTable.Angle_KMinMag1[membrane & poss])
            aKMinMag2 = np.array(FibersTable.Angle_KMinMag2[membrane & poss])
            ang_pos = np.array([aKMax[0], aKMin[0], aKMinMag1[0], aKMinMag2[0]])
            ang_mod = aloc[0]*np.ones(4)
            ax.plot(ang_mod, np.sort(ang_pos), linestyle='-.', label=pos)
#        for jj in range(4):
#            str1 = pos + angNam[jj]
#            ax2.plot(ang_mod[jj], ang_pos[jj], color=colrs[jj], marker=m, label=str1)
    ax.legend()
    
   
#%% load the results data: 
myPath = '/Users/df/Documents/myGits/FibersResultsXLSX/'
Top_file = myPath + 'Curvature_and_Directionality_Results_T.xlsx'

# convert the xlsx file to a Pandas dataFrame
FibersT = pd.read_excel(Top_file, index_col=None)

plot_PDF_1VM_b( FibersT )
plot_PDF_2VM_b( FibersT )

#%% Bottom: 
Bottom_file = myPath + 'Curvature_and_Directionality_Results_B.xlsx'
FibersB = pd.read_excel(Bottom_file, index_col=None)
plot_PDF_1VM_b( FibersB )
plot_PDF_2VM_b( FibersB )

#%% Right: 
Right_file = myPath + 'Curvature_and_Directionality_Results_R.xlsx'
FibersR = pd.read_excel(Right_file, index_col=None)
plot_PDF_1VM_b( FibersR )
plot_PDF_2VM_b( FibersR )

#%% Left: 
Left_file = myPath + 'Curvature_and_Directionality_Results_L.xlsx'
FibersL = pd.read_excel(Left_file, index_col=None)
plot_PDF_1VM_b( FibersL )
plot_PDF_2VM_b( FibersL )

#%% Center: 
Center_file = myPath + 'Curvature_and_Directionality_Results_C.xlsx'
FibersC = pd.read_excel(Center_file, index_col=None)
plot_PDF_1VM_b( FibersC )
plot_PDF_2VM_b( FibersC )

#%% alpha vs dispersion:
# load the results data: 
myPath = '/Users/df/Documents/myGits/FibersResultsXLSX/'
#ALL_file = myPath + 'Curvature_and_Directionality_Results_ALL.xlsx'
ALL_file = myPath + 'Processing.xlsx'

FibersALL = pd.read_excel(ALL_file, index_col=None)

#Position = FibersALL.Position[FibersALL.Position=='Top']
Position = ('Top','Bottom','Right','Left','Center')
plot_alpha_disp( FibersALL, Position )

## or if you prefere individually: 
#plot_alpha_disp( FibersALL, Position=('Top','') )
#plot_alpha_disp( FibersALL, Position=('Bottom','') )
#plot_alpha_disp( FibersALL, Position=('Right','') )
#plot_alpha_disp( FibersALL, Position=('Left','') )
#plot_alpha_disp( FibersALL, Position=('Center','') )

#plot_curvratio_disp( FibersALL, Position,  )

#%%
temp_path, im_name = os.path.split(ALL_file)
im_name_0 = os.path.splitext(ALL_file)[0]
xtypes = ("ratio", "beta", "angratio" )
ytypes = ("concentration", "dispersion", "normconc")
for xty in xtypes:
    for yty in ytypes:
        FIG, AxFIG = plot_curvratio_disp( FibersALL, Position, xtype=xty, ytype=yty, th=100)
        fig_name_path = im_name_0 + '_' + xty + '_' + yty + '.png'
        FIG.savefig(fig_name_path)

#%% per membrane, to correlate the angles: 

Position = ('Top','Bottom','Right','Left','Center')
for pos in Position:
    plot_angles_1VMU( FibersALL, Position=pos )

#%% or scatter: 

Position = ('Top','Bottom','Right','Left','Center')
for pos in Position:
    scatter_angles_1VMU( FibersALL, Position=pos )

#%% per membrane: 

rwms = pd.unique(FibersALL.Name)
for rwm in pd.unique(FibersALL.Name):
    #print(rwm)
    scatter_angles_RWM( FibersALL, RWM=rwm )

#%% playing with Ordinary Least Squares (OLS) of stamodels module: 
#from statsmodels.formula.api import ols

x = np.linspace(-5, 5, 20)
np.random.seed(1)
y = -5 + 3*x + 4*np.random.normal(size=x.shape)
data = pd.DataFrame({'x':x, 'y':y})
model = ols("y ~ x", data).fit()
print(model.summary())

#%%
#import seaborn
seaborn.pairplot(data, vars=['x', 'y'], kind='reg')

seaborn.lmplot(y='y', x='x', data=data)

seaborn.lmplot(y='VM1_Conc', x='beta', data=FibersALL)

#%%
Position = ('Top','Bottom','Right','Left','Center')

for pos in Position:
    temp_DF = pd.DataFrame({'VM1_Conc': FibersALL.VM1_Conc[FibersALL.Position==pos], \
                       'beta': FibersALL.beta[FibersALL.Position==pos]})
    #temp_DF = temp_DF[['VM1_Conc', 'beta']]
    
    seaborn.lmplot(y='VM1_Conc', x='beta', data=temp_DF).set(xlim=(-1, 46), xticks=np.arange(0,50,5)).set(title=pos)
    temp_model = ols("VM1_Conc ~ beta", temp_DF).fit()
    print('----------------------- \n *********************** \n -----------------------')
    print('OLS results for ' + pos + ':\n', temp_model.summary())

#%%
#gdd = seaborn.PairGrid(FibersALL, y_vars=["VM1_Conc"], x_vars=["beta", "Position"], size=4)
#gdd.map(seaborn.regplot, color=".3")
#gdd.set(ylim=(-1, 11), yticks=[0, 5, 10]);
#
#gsd = seaborn.PairGrid(FibersALL, y_vars=["VM1_Conc"], x_vars=["beta"], size=4).map(seaborn.regplot, color=".3")
#
seaborn.lmplot(y='VM1_Conc', x='beta', data=FibersALL, hue='Position')
# to take a quick look at a dataset, it can be easier to use pairplot(): 
seaborn.pairplot(FibersALL, vars=['beta', 'VM1_Conc'], kind='reg')
#seaborn.pairplot(FibersALL, hue="Position", size=2.5);
#
seaborn.PairGrid(FibersALL, y_vars=["VM1_Conc"], x_vars=["beta", "beta"], hue='Position', size=4).map(seaborn.regplot, color=".3")
#
seaborn.PairGrid(FibersALL, y_vars=["VM1_Conc"], x_vars=["beta", "beta"], hue='Position', size=4).map(seaborn.regplot )
#
seaborn.lmplot(y='Angle_KMinMag1', x='VM1_Ang', data=FibersALL, hue='Position')
seaborn.lmplot(y='Angle_KMinMag1', x='VM1_Ang', data=FibersALL)
#
seaborn.PairGrid(FibersALL, y_vars=['VM1_Ang'], x_vars=['Angle_KMax', 'Angle_KMin', 'Angle_KMinMag1', 'Angle_KMinMag2', 'VM1_Ang'], hue='Position', size=4).map(seaborn.regplot ).add_legend();
seaborn.pairplot(FibersALL, vars=['VM1_Ang', 'Angle_KMax', 'Angle_KMin', 'Angle_KMinMag1', 'Angle_KMinMag2'], kind='reg', hue='Position')
seaborn.pairplot(FibersALL, vars=['VM1_Ang', 'Angle_KMax', 'Angle_KMin', 'Angle_KMinMag1', 'Angle_KMinMag2'], kind='reg')
seaborn.PairGrid(FibersALL, y_vars=['VM1_Ang'], x_vars=['Angle_KMax', 'Angle_KMin', 'Angle_KMinMag1', 'Angle_KMinMag2', 'VM1_Ang'], hue='Position', size=4).map(plt.scatter, s=50, edgecolor="white" ).add_legend();

#%%
## the 1VMU model: 
#locs = np.radians(FibersT.Ang_1VM)
#kaps = FibersT.Disp_1VM
#wvm = FibersT.Weig_1VM
#wu = FibersT.Weigu_1VM
#
#fig1VMU, ax_1VMU = plot_PDF_1VM( locs, kaps, wvm, wu, Position=Position )
#
#Angle_KMax = FibersT.Angle_KMax
#Angle_KMin = FibersT.Angle_KMin
#Angle_KMinMag1 = FibersT.Angle_KMinMag1
#Angle_KMinMag2 = FibersT.Angle_KMinMag2
#yones = np.ones(len(Angle_KMax))
#
#ax_1VMU.plot(Angle_KMax, yones, 'o', color='black', label='KMax' )
#ax_1VMU.plot(Angle_KMin, 0.9*yones, 's', color='blue', label='KMin' )
#ax_1VMU.plot(Angle_KMinMag1, 0.8*yones, '^', color='red', label='MM1' )
#ax_1VMU.plot(Angle_KMinMag2, 0.7*yones, 'v', color='green', label='MM2' )
#
#ax_1VMU.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), \
#          fancybox=True, shadow=True, ncol=6)
#
##%%
## the 2VMU model: 
#locs1 = np.radians(FibersT.Ang1_2VM)
#kaps1 = FibersT.Disp1_2VM
#wvm1 = FibersT.Weig1_2VM
#locs2 = np.radians(FibersT.Ang2_2VM)
#kaps2 = FibersT.Disp2_2VM
#wvm2 = FibersT.Weig2_2VM
#wu = FibersT.Weigu_2VM
#
#plot_PDF_2VM( locs1, kaps1, wvm1, locs2, kaps2, wvm2, wu, Position=Position )

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:53:34 2018

@author: DF
"""

#import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import stats
import pandas as pd

# import packages specific to our work: 
import dff_dispersionCalculator as dC
import dff_mle_minimize_mvMU as DFmle
import dff_StatsTools as DFST

import os, os.path


# Define the path where the data from the FFT (the .csv file) is located: 
# myPath = '/Users/df/0_myDATA/testSamples/'
myPath = '/Users/df/Documents/myGits/fiberDirectionality/testSamples/'

# ------------------------------------------------------------------------- #
#    im_path = myPath + 'MAX_20X_Airyscan_6.jpg'
##    csv_path = myPath + "MAX_20X_Airyscan_6.csv"
#    csv_path = myPath + "MAX_20X_Airyscan_6_90.csv"
    
#    im_path = myPath + 'MAX_20180223_S2_20X_2c_CenterCropped3.png'
#    csv_path = myPath + "MAX_20180223_S2_20X_2c_CenterCropped3_FFT.csv"
    
im_path = myPath + 'Clipboard.png'
csv_path = myPath + "Dir_hist_FFT.csv"
#    csv_path = myPath + "Dir_hist_FFT_180.csv"

#    im_path = myPath + 'InCos.png'
#    csv_path = myPath + "InCos0pi.csv"

temp_path, temp_file = os.path.split(im_path)
print(temp_path)
print(temp_file)
# ------------------------------------------------------------------------- #
    
# read the data from the csv file: 
angles, values, mydat = dC.imageCVS2Data( csv_path )

# convert the angles from degrees to radians:
angles = angles
r_X = np.radians( angles )
X_samples = r_X

# normalize the light intensity data (FFT): 
n_X = dC.normalizeIntensity( angles, values, YNplot='Yes' )
Int = n_X[:,1]

# make points of angles based on the light intensity: 
p_X = dC.makePoints( n_X, YNplot='Yes' )

# select model to fit: 
#model_test = '1vM'
#model_test = '2vM'
#model_test = '3vM'
#model_test = '2vM1U'
model_test = '1vM1U'

# CAUTION: sensitive to the initial guess! so decide first which values 
#           to use based on the histogram of the original data.
#%% for a 3-vM model: 
#model_test = '3vM'
if model_test == '3vM':
    # parameters for the von Mises member:
    p1_ = 0.4                   # weight contribution of the 1st von Mises 
    p3_ = 0.3                   # weight contribution of the 3rd von Mises 
    p2_ = 1. - p1_ - p3_        # weight contribution of the 2nd von Mises 
    kappa1_ = np.array((12.0))  # concentration for the 1st von Mises member 
    kappa2_ = np.array((5.0))   # concentration for the 2nd von Mises member 
    kappa3_ = np.array((12.0))  # concentration for the 3rd von Mises member 
    loc1_ = -np.pi/3.0          # location for the 1st von Mises member 
    loc2_ = -0.*np.pi           # location for the 2nd von Mises member 
    loc3_ =  np.pi/3.0          # location for the 3rd von Mises member 
    
    # ------------------------------------- # 
    # if you solve with minimize: 
    in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_, p3_, kappa3_, loc3_ ]
    # bound constraints for the variables: 
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    # for the weight: (0., 1.)
    # for the concentration: (0., 100.)
    # for the location: (lim_l, lim_u)
    bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
            (0., 1.), (0., 100.), (lim_l, lim_u), \
            (0., 1.), (0., 100.), (lim_l, lim_u))
    # the sum of the weights should be equal to unity: 
    cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] + x[6] - 1.0})
    results = optimize.minimize(DFmle.logLik_3vM, in_guess, args=(r_X,Int), \
                                method='SLSQP', bounds=bnds, constraints=cons, \
                                tol=1e-6, options={'maxiter': 100, 'disp': True})
    print('METHOD II = ',results.x)
    print('-----------------------------------------')
    p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle, \
                                 p3_mle, kappa3_mle, mu3_mle = results.x
    print('p1, kappa1, mu1, p2, kappa2, mu2, p2, kappa2, mu2 = ',results.x)
    res = results.x
    # ------------------------------------- #   
    # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
    #                  [p2_mle, kappa2_mle, mu2_mle],
    #                  [p3_mle, kappa3_mle, mu3_mle]])
    
    # ---------------------------------------------------------------------- #
    # Store the results to arrays for later use:
    members_list = ["1st von Mises", "2nd von Mises", "3rd von Mises"]
    loc_mle = np.array([mu1_mle, mu2_mle, mu3_mle])
    kap_mle = np.array([kappa1_mle, kappa2_mle, kappa3_mle])
    p_mle = np.array([p1_mle, p2_mle, p3_mle])

#%% for a 2-vM model: 
#model_test = '2vM'
if model_test == '2vM':
    # parameters for the von Mises member:
    p1_ = 0.4                   # weight contribution of the 1st von Mises 
    p2_ = 1. - p1_              # weight contribution of the 2nd von Mises 
    kappa1_ = np.array((5.0))   # concentration for the 1st von Mises member 
    kappa2_ = np.array((12.0))  # concentration for the 2nd von Mises member 
    loc1_ = -np.pi/9.0          # location for the 1st von Mises member 
    loc2_ = 0.*np.pi/9.0        # location for the 2nd von Mises member 
    
    # ------------------------------------- # 
    # if you solve with minimize: 
    in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_ ]
    # bound constraints for the variables: 
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    # for the weight: (0., 1.)
    # for the concentration: (0., 100.)
    # for the location: (lim_l, lim_u)
    bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
            (0., 1.), (0., 100.), (lim_l, lim_u))
    # the sum of the weights should be equal to unity: 
    cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] - 1.0})
    results = optimize.minimize(DFmle.logLik_2vM, in_guess, args=(r_X,Int), \
                                method='SLSQP', bounds=bnds, constraints=cons, \
                                tol=1e-6, options={'maxiter': 100, 'disp': True})
    print('METHOD II = ',results.x)
    print('-----------------------------------------')
    p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle = results.x
    print('p1, kappa1, mu1, p2, kappa2, mu2 = ',results.x)
    res = results.x
    # ------------------------------------------------ #   
    # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
    #                  [p2_mle, kappa2_mle, mu2_mle]])
    
    # ---------------------------------------------------------------------- #
    # Store the results to arrays for later use:
    members_list = ["1st von Mises", "2nd von Mises"]
    loc_mle = np.array([mu1_mle, mu2_mle])
    kap_mle = np.array([kappa1_mle, kappa2_mle])
    p_mle = np.array([p1_mle, p2_mle])
    
#%% for a 2vM1U model: 
#model_test = '2vM1U'
if model_test == '2vM1U':
    # parameters for the von Mises member:
    p1_ = 0.4                   # weight contribution of the 1st von Mises 
    p2_ = 0.4                   # weight contribution of the 2nd von Mises 
    pu_ = 1. - p1_ - p2_        # weight contribution of Uniform distribution 
    kappa1_ = np.array((12.0))  # concentration for the 1st von Mises member 
    kappa2_ = np.array((5.0))   # concentration for the 2nd von Mises member 
    loc1_ = -np.pi/9.0          # location for the 1st von Mises member 
    loc2_ = -0.*np.pi/20.0      # location for the 2nd von Mises member 
    
    # ------------------------------------- # 
    # if you solve with minimize: 
    in_guess = [ p1_, kappa1_, loc1_, p2_, kappa2_, loc2_, pu_ ]
    # bound constraints for the variables: 
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    # for the weight: (0., 1.)
    # for the concentration: (0., 100.)
    # for the location: (lim_l, lim_u)
    bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
            (0., 1.), (0., 100.), (lim_l, lim_u), \
            (0., 1.))
    # the sum of the weights should be equal to unity: 
    cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] + x[6] - 1.0})
    results = optimize.minimize(DFmle.logLik_2vM1U, in_guess, args=(r_X,Int), \
                                method='SLSQP', bounds=bnds, constraints=cons, \
                                tol=1e-6, options={'maxiter': 100, 'disp': True})
    print('METHOD II = ',results.x)
    print('-----------------------------------------')
    p1_mle, kappa1_mle, mu1_mle, p2_mle, kappa2_mle, mu2_mle, \
                                                     pu_mle = results.x
    print('p1, kappa1, mu1, p2, kappa2, mu2, pu = ',results.x)
    res = results.x
    # ------------------------------------- #   
    # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
    #                  [p2_mle, kappa2_mle, mu2_mle],
    #                  [pu_mle, 0.0,        0.0]])
    
    # ---------------------------------------------------------------------- #
    # Store the results to arrays for later use:
    members_list = ["1st von Mises", "2nd von Mises", "Uniform"]
    loc_mle = np.array([mu1_mle, mu2_mle, 0.0])
    kap_mle = np.array([kappa1_mle, kappa2_mle, 1e-3])
    p_mle = np.array([p1_mle, p2_mle, pu_mle])
    
#%% for a 1-vM model: 
#model_test = '1vM'
if model_test == '1vM':
    # parameters for the von Mises member:
    p1_ = 1.0             # weight contribution of the 1st von Mises 
    kappa1_ = np.array((6.0))  # concentration for the 1st von Mises member 
    loc1_ = -np.pi/3.0           # location for the 1st von Mises member 
    
    # ------------------------------------- # 
    # if you solve with minimize: 
    in_guess = [ kappa1_, loc1_ ]
    # bound constraints for the variables: 
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    # for the weight: (0., 1.)
    # for the concentration: (0., 100.)
    # for the location: (lim_l, lim_u)
    bnds = ((0., 100.), (lim_l, lim_u))
    results = optimize.minimize(DFmle.logLik_1vM, in_guess, args=(r_X,Int), \
                                method='SLSQP', bounds=bnds, \
                                tol=1e-6, options={'maxiter': 100, 'disp': True})
    print('METHOD II = ',results.x)
    print('-----------------------------------------')
    kappa1_mle, mu1_mle = results.x
    print('kappa1, mu1 = ',results.x)
    res = results.x
    # ------------------------------------------------ #   
    # data = np.array([[p1_mle, kappa1_mle, mu1_mle]])
    
    # ---------------------------------------------------------------------- #
    # Store the results to arrays for later use:
    members_list = ["1st von Mises"]
    loc_mle = np.array([mu1_mle])
    kap_mle = np.array([kappa1_mle])
    p_mle = np.array([p1_])
    
#%% for a 1vM1U model: 
#model_test = '1vM1U'
if model_test == '1vM1U':
    # parameters for the von Mises member:
    p1_ = 0.7                   # weight contribution of the 1st von Mises 
    pu_ = 1. - p1_              # weight contribution of Uniform distribut 
    kappa1_ = np.array((12.0))  # concentration for the 1st von Mises member 
    loc1_ = -np.pi/9.0          # location for the 1st von Mises member 
    
    # ------------------------------------- # 
    # if you solve with minimize: 
    in_guess = [ p1_, kappa1_, loc1_, pu_ ]
    # bound constraints for the variables: 
    lim_l = -np.pi/2.
    lim_u =  np.pi/2.
    # for the weight: (0., 1.)
    # for the concentration: (0., 100.)
    # for the location: (lim_l, lim_u)
    bnds = ((0., 1.), (0., 100.), (lim_l, lim_u), \
            (0., 1.))
    # the sum of the weights should be equal to unity: 
    cons = ({'type': 'eq', 'fun': lambda x:  x[0] + x[3] - 1.0})
    results = optimize.minimize(DFmle.logLik_1vM1U, in_guess, args=(r_X,Int), \
                                method='SLSQP', bounds=bnds, constraints=cons, \
                                tol=1e-6, options={'maxiter': 100, 'disp': True})
    print('METHOD II = ',results.x)
    print('-----------------------------------------')
    p1_mle, kappa1_mle, mu1_mle, pu_mle = results.x
    print('p1, kappa1, mu1, pu = ',results.x)
    res = results.x
    # ------------------------------------- #   
    # data = np.array([[p1_mle, kappa1_mle, mu1_mle],
    #                  [pu_mle, 0.0,        0.0]])
    
    # ---------------------------------------------------------------------- #
    # Store the results to arrays for later use:
    members_list = ["1st von Mises", "Uniform"]
    loc_mle = np.array([mu1_mle, 0.0])
    kap_mle = np.array([kappa1_mle, 1e-3])
    p_mle = np.array([p1_mle, pu_mle])
    
# -------------------------------------------------------------------------- #
#%% COLLECT the estimated parameters into a Pandas dataFrame: 
loc_mle_d = np.degrees(loc_mle)
dataFrame = pd.DataFrame({'Distribution': members_list, \
                          'Weight': p_mle.ravel(), \
                          'Concentration': kap_mle.ravel(), \
                          'Location': loc_mle.ravel(), \
                          'Location (deg)': loc_mle_d.ravel()})
dataFrame = dataFrame[['Distribution', 'Weight', \
                       'Concentration','Location', 'Location (deg)']]
#dataFrame.set_index('Distribution')
print(dataFrame)

# -------------------------------------------------------------------------- #
#%% ------------------------------------------------------------------- # 
# PLOT in the same figure the original histogram and the model PDF: 
scal_ = 0.5
fig, ax = plt.subplots(1, 1, figsize=(9,4))
ax.set_title('Probability Density Function of Mixture Model (von Mises + Uniform)')
ax.plot(angles, Int, 'b-', label='Original data')
# prepare the PDFs: 
x_ = np.linspace( min(X_samples), max(X_samples), len(r_X) )
r_ = np.degrees(x_)
x_ax = r_
X_tot = np.zeros(len(x_),)
cXtot = np.zeros(len(x_),)
jj = 0
# plot in the same histogram the approximations:
for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
    jj += 1
    fX_temp = stats.vonmises( kap, mu, scal_ )
    # X_temp = pii*stats.vonmises.pdf( x_, kap, mu, scal_ )
    X_temp = pii*fX_temp.pdf ( x_ )
    X_tot += X_temp
#    ax.plot(x_ax, X_temp, linewidth=2, linestyle='--', \
#            label='von Mises member {} '.format(jj))
            #label=r'$\mu$ = {}, $\kappa$= {}, p= {} '.format(round(mu,3), round(kap,3), round(pii,3)))
    # this is wrong!!!: 
    cXtot += pii*fX_temp.cdf( x_ )
    # cXtot += pii*stats.vonmises.cdf( x_, kap, mu, scal_ )
    
ax.plot(x_ax, X_tot, color='red', linewidth=2, linestyle='-', label='Mixture fit')
ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
ax.set_ylabel(r'$f(\theta)$', fontsize=12)
ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
ax.legend(loc=1)
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
#          fancybox=True, shadow=True, ncol=3)


# -------------------------------------------------------------------------- #
#%% ------------------------------------------------------------------- #
# PREPARE THE DATA FOR THE GOODNESS-OF-FIT TESTS: 
# based on populated points for every angle based on its intensity: 
# CAUTION !!!
#   THIS IS THE CORRECT APPROACH !!!
aa = np.sort( p_X )
aad = np.degrees(aa)
fX_t = np.zeros(len(aa),)
cX_t = np.zeros(len(aa),)
for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
    temp = stats.vonmises( kap, mu, scal_ )
    fX_t += pii*temp.pdf( aa )
    cX_t += pii*temp.cdf( aa )
    #fX_t += pii*stats.vonmises.pdf( aa, kap, mu, scal_ )
    #cX_t += pii*stats.vonmises.cdf( aa, kap, mu, scal_ )

x1, cfX_obs = DFST.ECDF( p_X )
x1d = np.degrees(x1)

if max(cX_t) > 1:
    cX_c = cX_t - (max(cX_t) - 1.)
else:
    cX_c = cX_t

# plot the CDF from populated points: 
fig, ax = plt.subplots(1,1,figsize=(4,3))
ax.set_title('CDF of Mixture Model')
ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
ax.set_ylabel('Cumulative distribution', fontsize=12)
ax.plot( x1d, cfX_obs, 'b-', lw=2, alpha=0.6, label='ECDF data')
ax.plot( aad, cX_c, 'r--', label='CDF model')
ax.legend()

# -------------------------------------------------------------------------- #
#%% GOF tests: 

# The Watson GOF: 
U2, Us, uc, pu2, pus, H0_W2, lev_W2 = DFST.watson_GOF( cX_c, alphal=1)
print('Watson GOF =',U2, Us, uc, pu2, pus, H0_W2, lev_W2)

    
# The Kuiper GOF: 
# this is the most correcrt: 
Vn, Vc, pVn, H0_K2, lev_K2 = DFST.Kuiper_GOF( cX_c )
print('Kuiper GOF =', Vn, pVn, H0_K2, lev_K2)


# The R2 coefficient: 
fX_r2 = np.zeros(len(x1),)
for mu, kap, pii in zip(loc_mle, kap_mle, p_mle):
    temp = stats.vonmises( kap, mu, scal_ )
    fX_r2 += pii*temp.pdf( x1 )
    
dx = np.diff(x1)
cX_r2 = np.ones(len(x1),)
cX_r2[0:-1] = np.cumsum(fX_r2[0:-1]*dx)
R2 = DFST.myR2( cX_r2, cfX_obs )
print('R2 =', R2)
if R2 > 0.90:
    H0_R2 = "Do not reject"
else:
    H0_R2 = "Reject"
    
# Plot the Probability-Probability Plot: 
DFST.PP_GOF( cX_r2, cfX_obs )

# ---------------------------------------------------------------------- #
#%% Collect all results into a Pandas dataFrame: 
gof_list = ["Waston", "Kuiper", "R2"]
gof_stats = np.array([U2, Vn, R2])
gof_crVal = np.array([uc, Vc, 1.0])
gof_pvals = np.array([pu2, pVn, 1.0])
gof_level = np.array([lev_W2, lev_K2, 1.0])
gof_rejH0 = [H0_W2, H0_K2, H0_R2]

dataFrameGOF = pd.DataFrame({'GOF': gof_list, \
                             'Statistic': gof_stats.ravel(), \
                             'CriticalValue': gof_crVal.ravel(), \
                             'PValue': gof_pvals.ravel(), \
                             'SignLev': gof_level.ravel(), \
                             'Decision': gof_rejH0})
dataFrameGOF = dataFrameGOF[['GOF', 'Statistic', \
                             'CriticalValue','PValue', 'SignLev', 'Decision']]
print(dataFrameGOF)

 

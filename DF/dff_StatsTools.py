#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:47:42 2018
Author Dimitrios Fafalis 

Contents of the package:
    
    Import this package as: DFST
    
    Tests for Goodness-of-fit: 
    -------------------------
    1. PP_GOF
    2. myR2
    3. ECDF_Intensity
    4. ECDF
    5. watson_CDF
    6. watson_GOF
    7. Kuiper_GOF
    8. Kuiper_CDF
    
    - CAUTION: DO NOT USE THE FOLLOWING 4 (FOR NOW)
    0. KS_CDF
    0. my_KS_GOF_mvM
    0. my_KS_GOF_mvM_I
    0. my_chisquare_GOF
    - 
    
    Generate random samples and plot: 
    --------------------------------
    1. rs_mix_vonMises
    2. rs_mix_vonMises2
    3. plot_rv_distribution
    4. plot_mixs_vonMises_Specific

@author: df
"""

from scipy import stats
from scipy.stats import vonmises

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

    
# ------------------------------------------------------------------------- #
def PP_GOF( Fexp, Fobs ):
    """
    Generate the Probability-Probability plot.
    Fexp:   the distribution function of the postulated (expected) distribution
    Fobs:   the empirical distribution function (the original data)
    """
    x = np.linspace(0,1,len(Fobs))
    
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.plot(Fexp, Fobs, 'k.', lw=2, alpha=0.6, label='P-P plot')
    ax.plot(x, x, 'r--', lw=2, alpha=0.6, label='1:1')
    ax.set_title('P-P plot for a Mixture Model')
    ax.set_xlabel('Mixture distribution function')
    ax.set_ylabel('Empirical distribution function')
    ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
    ax.legend()
    
# ------------------------------------------------------------------------- #
def myR2( Fexp, Fobs ):
    """
    Function to compute the coefficient of determination R2, 
    a measure of goodness-of-fit. 
    Fobs: empirical (observed) CDF,
            can be computed from the function: 
                ECDF_Intensity( angles, values )
    Fexp: expected (postulated) CDF
    Returns: R2 
    """
    
    Fbar = np.mean(Fexp)
    
    FobsmFexp = sum((Fobs - Fexp)**2)
    
    num = sum((Fexp - Fbar)**2)
    
    den = num + FobsmFexp
    
    R2 = num / den
        
    return R2

# ------------------------------------------------------------------------- #
def ECDF_Intensity( angles, values ):
    """
    Function to evaluate the ECDF of raw data given in the form of intensity.
    angles:     equally-spaced angles (rads) from the FFT of an image
    values:     the intensity of light at every radial segment of angles. 
    Returns:    x:          the angles in rads
                ECDF_I:    the CDF of the observed intensity over angles.
    """
    x = np.radians( angles )
    dx = np.radians( abs(angles[0] - angles[1]) )
    y_intQ = np.trapz( values, x=np.radians(angles) )
    n_X = values/y_intQ
    ECDF_I = np.cumsum( n_X*dx )
    
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    ax.plot(x, ECDF_I, 'k', lw=3, alpha=0.6, label='ECDF data')
    ax.set_title('Plot from ECDF_Intensity')
    ax.legend()
    
    return x, ECDF_I

# ------------------------------------------------------------------------- #
def ECDF( data ):
    """
    CAUTION!: NOT for Light Intensity (FFT) data! 
    Function to evaluate the empirical distribution function (ECDF) 
    associated with the empirical measure of a sample. 
    data:   random sample, (rads) in the nature of the problem to consider 
    Returns:    x_values:   the x-axis of the ECDF figure (rads)
                y_values:   the y-axis of the ECDF figure, the ECDF
    """
    raw_data = np.array(data)
    # create a sorted series of unique data
    cdfx = np.sort(np.unique(raw_data))
#    raw_data = data
#    # create a sorted series of unique data
#    cdfx = np.sort(data)
    # x-data for the ECDF: evenly spaced sequence of the uniques
    x_values = np.linspace(start=min(cdfx),stop=max(cdfx),num=len(cdfx))
    # size of the x_values
    size_data = raw_data.size
    # y-data for the ECDF:
    y_values = []
    for i in x_values:
        # all the values in raw data less than the ith value in x_values
        temp = raw_data[raw_data <= i]
        # fraction of that value with respect to the size of the x_values
        value = temp.size / size_data
        # pushing the value in the y_values
        y_values.append(value)
        
    # return both x and y values
    return x_values, y_values

# ------------------------------------------------------------------------- #
def watson_CDF( x ):
    """
    Function to return the value of the CDF of the asymptotic Watson 
    distribution at x: 
    """
    epsi = 1e-10
            
    k = 1
    y = 0
    asum = 1
    while asum > epsi:
        md = ((-1)**(k-1))*np.exp(-2.*((np.pi)**2)*(k**2)*x)
        y += 2.*md
        k += 1
        asum = abs(md)
        
    print('k iterations =',k)
    return y

# ------------------------------------------------------------------------- #
def watson_GOF( cdfX, alphal=2):
    """
    From Book: "Applied Statistics: Using SPSS, STATISTICA, MATLAB and R"
                by J. P. Marques de Sa, Second Edition
                SPringer, ISBN: 978-3-540-71971-7 
                Chapter 10-Directional Statistics, Section 10.4.3 
    Watson U2 test of circular data A.
    cdfX is the EXPECTED distribution (cdf).
    U2 is the Watson's test statistic
    US is the modified test statistic for known (loc, kappa)
    UC the critical value at ALPHAL (n=100 if n>100)
    ALPHAL: 1=0.1; 2=0.05 (default); 3=0.025; 4=0.01; 5=0.005
    """
    
    n = len(cdfX)
    V = cdfX
    Vb = np.mean(V)
    cc = np.arange(1., 2*n, 2)
    
    # The Watson's statistic: 
    U2 = np.dot(V, V) - np.dot(cc, V)/n + n*(1./3. - (Vb - 0.5)**2.)
    
    # The modified Watson's statistic (when both loc and kappa are known): 
    Us = (U2 - 0.1/n + 0.1/n**2)*(1.0 + 0.8/n)
    
    alpha_lev = np.array([0.1, 0.05, 0.025, 0.01, 0.005])
    #              alpha=   =0.1     =0.05    =0.025   =0.01    =0.005
    c = np.array([ [ 2,	0.143,	0.155,	0.161,	0.164,	0.165 ],
                   [ 3,	0.145,	0.173,	0.194,	0.213,	0.224 ],
                   [ 4,	0.146,	0.176,	0.202,	0.233,	0.252 ],
                   [ 5,	0.148,	0.177,	0.205,	0.238,	0.262 ],
                   [ 6,	0.149,	0.179,	0.208,	0.243,	0.269 ],
                   [ 7,	0.149,	0.180,   0.210,	0.247,	0.274 ],
                   [ 8,	0.150,	0.181,	0.211,	0.250,	0.278 ],
                   [ 9,	0.150,	0.182,	0.212,	0.252,	0.281 ],
                   [ 10,	0.150,	0.182,	0.213,	0.254,	0.283 ],
                   [ 12,	0.150,	0.183,	0.215,	0.256,	0.287 ],
                   [ 14,	0.151,	0.184,	0.216,	0.258,	0.290 ],
                   [ 16,	0.151,	0.184,	0.216,	0.259,	0.291 ],
                   [ 18,	0.151,	0.184,	0.217,	0.259,	0.292 ],
                   [ 20,	0.151,	0.185,	0.217,	0.261,	0.293 ],
                   [ 30,	0.152,	0.185,	0.219,	0.263,	0.296 ],
                   [ 40,	0.152,	0.186,	0.219,	0.264,	0.298 ],
                   [ 50,	0.152,	0.186,	0.220,	0.265,	0.299 ],
                   [ 100,	0.152,	0.186,	0.221,	0.266,	0.301 ] ])

    # compute the critical value, by linear interpolation: 
    if n >= 100:
        uc = c[-1, alphal]
    else:
        for i in np.arange(0, len(c)): 
            if c[i, 0] > n:
                break
        n1 = c[i-1, 0]
        n2 = c[i, 0]
        c1 = c[i-1, alphal]
        c2 = c[i, alphal]
        uc = c1 + (n - n1)*(c2 - c1)/(n2 - n1)
    
    # compute the p-values: 
    pval2 = watson_CDF( U2 )
    pvals = watson_CDF( Us )
    
    if U2 < uc:
        H0_W = "Do not reject"
    else:
        H0_W = "Reject"
    
    return U2, Us, uc, pval2, pvals, H0_W, alpha_lev[alphal]

# ------------------------------------------------------------------------- #
def Kuiper_GOF( cY_ ):
    """
    cY_:  the probability distribution of the postulated model on ordered data
    """
    n = len(cY_)
    
    # prepare the Kuiper test: 
    ii = np.arange(0, 1, 1/n)
    vec1 = cY_ - ii
    jj = np.arange(1/n, 1+1/n, 1/n)
    vec2 = jj - cY_
    
    Dp = max(vec1)
    Dm = max(vec2)
    
    # compute the Kuiper statistic Vn: 
    Vn = (np.sqrt(n))*(Dp + Dm)
    # compute the Kuiper p-value: 
    pVn = Kuiper_CDF( Vn, n )
    
    # Significance levels: 
    alpha = np.array([0.10, 0.05, 0.025, 0.01])
    Vc = np.array([1.620, 1.747, 1.862, 2.001])
    if Vn < min(Vc):
        H0_K = "Do not reject"
        dif_v = abs(Vn - Vc)
        ind_v = np.argmin(dif_v)
        print('ind_v:',ind_v)
        alp_lev = alpha[ind_v]
        Vcc = Vc[ind_v]
    else:
        H0_K = "Reject"
        Vcc = Vc(-1)
    
    return Vn, Vcc, pVn, H0_K, alp_lev
    
# ------------------------------------------------------------------------- #
def Kuiper_CDF( x, n ):
    """
    Function to return the value of the CDF of the asymptotic Kuiper
    distribution at x: 
    """
    epsi = 1e-10
    
    x2 = x * x
    g = -8.*x/(3.*np.sqrt(n))
    k = 1
    y = 0
    asum = 1
    while asum > epsi:
        k2 = k * k
        md1 = (4.*k2*x2 - 1.)*np.exp(-2.*k2*x2)
        md2 = k2*(4.*k2*x2 - 3.)*np.exp(-2.*k2*x2)
        y += 2.*md1 + g*md2
        k += 1
        asum = abs(md1+md2)
        
    print('Kuiper k iterations =', k)
    return y

# ------------------------------------------------------------------------- #
# CAUTION: FOR NOW, DO NOT USE THE FOLLOWING TESTS:     
# ------------------------------------------------------------------------- #
def KS_CDF( x ):
    """
    Function to return the value of the CDF of the Kolmogorov-Smirnov
    distribution at x: 
    """
    epsi = 1e-10
    
    k = 1
    y = 1
    asum = 1
    xx = -2.0 * x * x;
    while asum > epsi:
        md = ((-1)**k)*np.exp(k * k * xx)
        y += 2.*md
        k += 1
        asum = abs(md)
        
    print('K-S k iterations =', k)
    return y
    
# ------------------------------------------------------------------------- #
def my_KS_GOF_mvM( X, Y_data, alpha=0.05 ):
    """
    KEEP this code!
    Function to get the GOF Kolmogorov-Smirnov test and its p-value
    CAUTION 1:  This function is for mixtures of von Mises distributions
    CAUTION 2:  X should be a random sample that can be related to a continuous 
                distribution
                It is NOT suitable for data like the INTENSITY of light we get
                from the FFT of an image: ->> NEEDS modification. 
    X:          the observed r.s., relevant to the nature of the problem 
                    they should be RANDOM!!! avoid  equally-spaced!
    Y_data:     a data frame containing the estimated parameters of the model
                the model could be a mixture of von Mises distributions only or
                a mixture of von Mises and Uniform distributions.
                By convention: the Uniform distribution, if any, is stored at
                            the last line of the dataFrame 'Y_data' 
    alpha:      the level of significance; default is 0.05
    """
    
    N = len(X)
    xx = np.sort(X)
    # CAUTION: NEVER USE EQUALLY-SPACED r.s. WITH K-S STATISTIC!!! 
    #xx = np.linspace(start=min(xx),stop=max(xx),num=len(xx))
    # the values of the theoretical (model) distribution for the r.s. X is: 
    scal_ = 0.5 # to scale the distribution on the semi-circle 
    kap_mle = np.array(Y_data.Concentration)
    loc_mle = np.array(Y_data.Location)
    w_mle = np.array(Y_data.Weight)
    cY_t = np.zeros(N,) # initialize the CDF 
    fY_t = np.zeros(N,) # initialize the PDF 
    for mu, kap, wi in zip(loc_mle, kap_mle, w_mle):
        # create a class of a single von Mises: 
        fY_i = stats.vonmises( kap, mu, scal_ )
        # take the pdf of the above single von Mises and add it to the model: 
        fY_t += wi*fY_i.pdf(xx)
        # take the cdf of the above single von Mises and add it to the model: 
        cY_t += wi*fY_i.cdf(xx)
        #cY_i = wi*stats.vonmises.cdf( xx, kap, mu, scal_ )
        #cY_t +=cY_i
    
    if max(cY_t) > 1:
        cY_c = cY_t - (max(cY_t) - 1.)
    elif min(cY_t) < 0:
        cY_c = cY_t + abs(min(cY_t))
    else:
        cY_c = cY_t
    
    # get the ECDF of the r.s.: 
    # CAUTION: the x_values returned by ECDF() are equally-spaced!!! 
    x_values, y_values = ECDF( X )
    
    # ------------------------------------- #
    # a. for the += case of computing the CDF: 
    # compute the K-S test: 
    cY_ = cY_c
#    ii = np.arange(0, 1+1/N, 1/N)
#    vec1 = cY_ - ii[0:-1]
#    vec2 = ii[1::] - cY_
    ii = np.arange(0, 1, 1/N)
    vec1 = cY_ - ii
    jj = np.arange(1/N, 1+1/N, 1/N)
    vec2 = jj - cY_
    # ------------------------------------- #
    
    # b. get the model CDF using cumsum: 
    # compute the K-S test: 
    dx = np.diff(xx)
    cY_b = np.ones(len(xx),)
    cY_b[0:-1] = np.cumsum(fY_t[0:-1]*dx)
#    cY_ = cY_b
#    ii = np.arange(0, 1, 1/N)
#    vec1 = cY_ - ii
#    jj = np.arange(1/N, 1+1/N, 1/N)
#    vec2 = jj - cY_
#    # ------------------------------------- #
    
    Dm = np.array([vec1,vec2])
    D = max( np.max(Dm,axis=0) )
    sqnD = (np.sqrt(N))*D
    print('sqrt(N))*D =', sqnD)
    pval = 1 - KS_CDF( sqnD )
    print('p-value:', pval)
    pval_mod = 1 - KS_CDF( sqnD + 1./(6.*np.sqrt(N)) )
    print('p-value-mod:', pval_mod)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    ax.set_title('From my_KS_GOF')
    ax.plot(xx, fY_t, 'b', lw=2, alpha=0.6, label='PDF fit')
    ax.plot(xx, cY_b, 'r', lw=2, alpha=0.6, label='CDF fit (cumsum)')
    ax.plot(xx, cY_c, 'g:', lw=2, alpha=0.6, label='CDF fit (+=)')
    ax.plot(x_values, y_values, 'c-.', lw=2, alpha=0.6, label='ECDF')
    ax.legend()
    
    # the scipy.kstest function returns the following: 
    #print(stats.kstest( X, pY_t ))
    #stats.kstest( Y5, 'vonmises', args=(s5[0], s5[1], scal_), alternative = 'greater' )
    
    # critical points: 
    d001 = 1.63/np.sqrt(N)
    d005 = 1.36/np.sqrt(N)
    d010 = 1.22/np.sqrt(N)
    # if you want to check, do this: 
    # P(1.63) = 1 - KS_CDF( 1.63 ) = 0.01
    # P(1.36) = 1 - KS_CDF( 1.36 ) = 0.05
    # P(1.22) = 1 - KS_CDF( 1.22 ) = 0.10
    crit_points = np.array([ d001, d005, d010 ])
    
    alpha_ = np.array([ 0.01, 0.05, 0.10 ])
    
    sig = pd.DataFrame({'alpha': alpha_, \
                        'crit_points': crit_points})
    print(sig)
    
    ind = sig.index[sig['alpha'] == alpha].tolist()[0]
    
    if D > sig.crit_points[ind]:
        H0 = 'reject'
    else:
        H0 = 'do not reject'
    
    KS_res = pd.DataFrame({'Statistic': 'Kolmogorov-Smirnov', 'D_N': [D], \
                           'p-value': pval, \
                           'critical value': sig.crit_points[ind], \
                           'alpha': [alpha], 'decision': H0})
    KS_res = KS_res[['Statistic', 'D_N', 'critical value', 'p-value', \
                     'alpha', 'decision']]
    
    print(KS_res)
    
    return KS_res



def my_KS_GOF_mvM_I( X, Y_data, alpha=0.05 ):
    """
    Function to get the GOF Kolmogorov-Smirnov test and its p-value
    CAUTION 1:  This function is for mixtures of von Mises distributions
    CAUTION 2:  X should be a random sample that can be related to a continuous 
                distribution
                It is NOT suitable for data like the INTENSITY of light we get
                from the FFT of an image: ->> NEEDS modification. 
    X:          the observed r.s., relevant to the nature of the problem 
    Y_data:     a data frame containing the estimated parameters of the model
                the model could be a mixture of von Mises distributions only or
                a mixture of von Mises and Uniform distributions.
                By convention: the Uniform distribution, if any, is stored at
                            the last line of the dataFrame 'Y_data' 
    alpha:      the level of significance; default is 0.05
    """
    angles = X[:,0]
    values = X[:,1]
    X = np.radians(angles)
    
    N = len(X)
    xx = np.sort(X)
    #xx = np.linspace(start=min(xx),stop=max(xx),num=len(xx))
    # the values of the theoretical (model) distribution for the r.s. X is: 
    scal_ = 0.5 # to scale the distribution on the semi-circle 
    kap_mle = np.array(Y_data.Concentration)
    loc_mle = np.array(Y_data.Location)
    w_mle = np.array(Y_data.Weight)
    cY_b = np.zeros(N,) # initialize the CDF 
    fY_t = np.zeros(N,) # initialize the PDF 
    for mu, kap, wi in zip(loc_mle, kap_mle, w_mle):
        fY_i = stats.vonmises( kap, mu, scal_ )
        fY_t += wi*fY_i.pdf(xx)
        # this may be wrong for the total CDF!!!: 
        cY_b += wi*fY_i.cdf(xx)
        #cY_i = wi*stats.vonmises.cdf( xx, kap, mu, scal_ )
        #cY_t +=cY_i
    
    # use the following only for the intensity values, since the angles are
    #   equally-spaced. Do not use for a r.s. generated by python. 
    #dx = abs(xx[0] - xx[1])
    
    # get the ECDF of the r.s.
    x_values, y_values = ECDF_Intensity( angles, values )
    
#    # ------------------------------------- #
#    # for the += case of computing the CDF: 
#    # compute the K-S test: 
#    cY_ = cY_b
#    ii = np.arange(0, 1+1/N, 1/N)
#    vec1 = cY_ - ii[0:-1]
#    vec2 = ii[1::] - cY_
#    # ------------------------------------- #
    
    # ------------------------------------- #
    # for the cumsum case of computing the CDF: 
    # compute the K-S test: 
    dx = np.diff(xx)
    cY_t = np.cumsum(fY_t[0:-1]*dx)
#    cY_ = cY_t
#    ii = np.arange(0, 1, 1/len(dx))
#    vec1 = cY_ - ii
#    jj = np.arange(1/len(dx), 1+1/len(dx), 1/len(dx))
#    vec2 = jj - cY_
#    # ------------------------------------- #
    
#    Dm = np.array([vec1,vec2])
#    D = max(np.max(Dm,axis=0))
    
    # compute the K-S test: 
    Dm = abs(cY_t - y_values[0:-1])
    D = max(Dm)
    
    dd = (np.sqrt(N))*D
    print('sqrt(N))*D =', dd)
    pval = 1 - KS_CDF( dd )
    print('p-value:', pval)
    pval_mod = 1 - KS_CDF( dd + 1./(6.*np.sqrt(N)) )
    print('p-value-mod:', pval_mod)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    ax.plot(xx, fY_t, 'b', lw=2, alpha=0.6, label='PDF fit')
    ax.plot(xx[0:-1], cY_t, 'r', lw=2, alpha=0.6, label='CDF fit (cumsum)')
    ax.plot(xx, cY_b, 'g:', lw=2, alpha=0.6, label='CDF fit (+=)')
    ax.plot(x_values, y_values, 'c-.', lw=2, alpha=0.6, label='ECDF')
    ax.set_title('From my_KS_GOF')
    ax.legend()
    
    # the scipy.kstest function returns the following: 
    #print(stats.kstest( X, pY_t ))
    #stats.kstest( Y5, 'vonmises', args=(s5[0], s5[1], scal_), alternative = 'greater' )
    
    # critical points: 
    d001 = 1.63/np.sqrt(N)
    d005 = 1.36/np.sqrt(N)
    d010 = 1.22/np.sqrt(N)
    crit_points = np.array([d001, d005, d010])
    
    alpha_ = np.array([0.01, 0.05, 0.10])
    
    sig = pd.DataFrame({'alpha': alpha_, \
                        'crit_points': crit_points})
    print(sig)
    
    ind = sig.index[sig['alpha'] == alpha].tolist()[0]
    
    if D > sig.crit_points[ind]:
        H0 = 'reject'
    else:
        H0 = 'do not reject'
    
    KS_res = pd.DataFrame({'Statistic': 'Kolmogorov-Smirnov', 'D_N': [D], \
                           'p-value': pval, \
                           'critical value': sig.crit_points[ind], \
                           'alpha': [alpha], 'decision': H0})
    KS_res = KS_res[['Statistic', 'D_N', 'critical value', 'p-value', \
                     'alpha', 'decision']]
    
    print(KS_res)
    
    return KS_res






def my_chisquare_GOF( X, Y_data, alpha=0.05 ):
    """
    Function to get the GOF chi-square test and its p-value
    X:      the observed r.s.
    Y_data: a data frame containing the estimated parameters of the model
            the model could be a mixture of von Mises distributions only or
            a mixture of von Mises and Uniform distributions.
            By convention: the Uniform distribution, if any, is stored at the 
                            last line of the dataFrame 'Y_data' 
    alpha:  the level of significance; default is 0.05
    """
    
    # the size of the data: 
    N = len(X)
    # determine the frequency bins: 
    nbin = 32
    nbb = 100
        
    # determine how many model parameters have been estimated by the data: 
    if any(Y_data.Distribution == 'Uniform'):
        c = 3*(len(Y_data)-1) + 1
    else:
        c = 3*len(Y_data)
    
    # the degrees of freedom for the chi-squared test: 
    ddf = nbin - 1 - c
    
    # plot the histogram of the data: 
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    ax.set_title('From my_chisquare_GOF')
    # get the frequencies on every bin, and the bins: 
    (nX1, bX1, pX1) = ax.hist(X, bins=nbin, density=False, \
                                            label='r.s.', color = 'skyblue' )
    print('# of frequencies per bin:', len(nX1))
    print('# of bins:', len(bX1))
    print('frequencies:', nX1)
    print('bins:', bX1)
    print('sum(nX1):',sum(nX1))
    
    # get the observed frequencies and compute the CDF for the observed data: 
    E1 = (np.cumsum(nX1))/sum(nX1)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    ax.set_title('From my_chisquare_GOF')
    (nXb, bXb, pXb) = ax.hist(X, bins=nbb, density=True, label='r.s.', color = 'skyblue' );
    ax.plot(bX1[0:-1], E1, 'b', label='CDF r.s.')

    # the observed frequencies: 
    O1 = nX1
    
    # the expected frequencies, after fitting the r.s. to a model: 
    scal_ = 0.5
    kap_mle = np.array(Y_data.Concentration)
    loc_mle = np.array(Y_data.Location)
    w_mle = np.array(Y_data.Weight)
    fY_t = np.zeros(len(bX1),)
    fY_b = np.zeros(len(bXb),)
    cY_t = np.zeros(len(bX1),)
    for mu, kap, wi in zip(loc_mle, kap_mle, w_mle):
        # a class member of a von Mises: 
        fY_i = stats.vonmises( kap, mu, scal_ )
        # the pdf of the member: 
        fY_t += wi*fY_i.pdf(bX1)
        # the cdf of the member: 
        # this may be wrong for the total CDF!!!: for some cases
        cY_i = wi*stats.vonmises.cdf( bX1, kap, mu, scal_ )
        cY_t += cY_i
        exp1 = N*np.diff(cY_t)
        fY_b += wi*fY_i.pdf(bXb)
        
    ax.plot(bX1, cY_t, 'r', label='CDF fit (+=)')
    
    dx = np.diff(bX1)
    cY_ = np.zeros(len(bX1),)
    cY_[0] = 0.0
    cY_[1::] = np.cumsum(fY_t[0:-1]*dx)
#    exp1 = N*np.diff(cY_)
#    print('len(exp1):',len(exp1))
    
    ax.plot(bX1, cY_, 'g', label='CDF fit (cumsum)')
    ax.legend()
    
    # get the chi-squared statistic based on the formula: 
    chi2 = np.sum(((O1 - exp1)**2)/exp1)
    #chi2 = np.sum(((abs(O1 - exp1) - 0.5)**2)/exp1)
    
    
    # Find the p-value: 
    # the effecrive dof: k - 1 - ddof 
    p_value = 1 - stats.chi2.cdf( x=chi2, df=ddf )  
    
    # get the critical value use k -1 - ddof 
    crit_val = stats.chi2.ppf( q = 1-alpha, df = ddf )
    
    # get the chi-square statistic from scipy.stats function: 
    # use as ddof the model parameters that are estimated from the sample: 
    stats_chi2 = stats.chisquare( f_obs=O1, f_exp=exp1, ddof=c )
    
    if chi2 > crit_val:
        H0 = 'reject'
    else:
        H0 = 'do not reject'
    if stats_chi2[0] > crit_val:
        H0t = 'reject'
    else:
        H0t = 'do not reject'
    
    statistic_ = ["My chi-square", "scipy.chisquare"]
    chi2_ = np.array([ chi2, stats_chi2[0] ])
    pval_ = np.array([ p_value, stats_chi2[1] ])
    cval_ = np.array([ crit_val, crit_val ])
    H0_ = (H0, H0t)
    chi2_results = pd.DataFrame({'Statistic': statistic_, \
                                 'chi^2': chi2_.ravel(), \
                                 'p-value': pval_.ravel(), \
                                 'critical value': cval_.ravel(), \
                                 'decision': H0_, \
                                 'alpha': alpha })
    chi2_results = chi2_results[['Statistic', 'chi^2', 'critical value', \
                                 'p-value', 'alpha', 'decision']]
    print(chi2_results)
    
    # ---------------------------------------------------------------------- #
    # NOT CORRECT !!! 
    # another test: use the pdfs in the formula: 
    chi2b = np.sum(((nXb - fY_b[0:-1])**2)/fY_b[0:-1])
    p_valueb = 1 - stats.chi2.cdf( x=chi2b, df=nbb-1-c )
    crit_valb = stats.chi2.ppf( q = 1-alpha, df = nbb-1-c )
    stats_chi2b = stats.chisquare( f_obs=nXb, f_exp=fY_b[0:-1], ddof=c )
    
    if chi2b > crit_valb:
        H0_b = 'reject'
    else:
        H0_b = 'do not reject'
    if stats_chi2b[0] > crit_valb:
        H0t_b = 'reject'
    else:
        H0t_b = 'do not reject'
    print('H0_b:',H0_b)
    print('H0t_b:',H0t_b)
    print('p_valueb:',p_valueb)
    print('chi2b:',chi2b)
    print('crit_valb:',crit_valb)
    print('stats_chi2b:',stats_chi2b)

    statistic_b = ["My chi-square", "scipy.chisquare"]
    chi2_b = np.array([ chi2b, stats_chi2b[0] ])
    pval_b = np.array([ p_valueb, stats_chi2b[1] ])
    cval_b = np.array([ crit_valb, crit_valb ])
    H0b_ = (H0_b, H0t_b)
    chi2_resultsb = pd.DataFrame({'Statistic': statistic_b, \
                                 'chi^2': chi2_b.ravel(), \
                                 'p-value': pval_b.ravel(), \
                                 'critical value': cval_b.ravel(), \
                                 'decision': H0b_, \
                                 'alpha': alpha })
    chi2_resultsb = chi2_resultsb[['Statistic', 'chi^2', 'critical value', \
                                 'p-value', 'alpha', 'decision']]
    print(chi2_resultsb)
    # ---------------------------------------------------------------------- #
    
    return chi2_results


def rs_mix_vonMises( kappas, locs, ps, sample_size=None ):
    """
    Generate random sample from a mixture of von Mises and/or Uniform 
        distributions, given their parameters. 
    Inspired by the web post: 
        "Creating a mixture of probability distributions for sampling"
            a question on stackoverflow:
                https://stackoverflow.com/questions/47759577/
                creating-a-mixture-of-probability-distributions-for-sampling
    """
    # number of von Mises distributions:
    num_distr = len(kappas)
    coefficients = ps
    coefficients /= coefficients.sum() # in case these did not add up to 1
    if len(kappas) < len(ps):
        # account for a uniform distribution: 
        num_distr +=1
        u1_ = -np.pi/2
        u2_ =  np.pi
        
    # to change the scale of the von Mises distributions: 
    scal_ = 0.5
    transfer_ = np.pi*scal_
#    kappa1_ = kappas[0]     # concentration for the 1st von Mises member 
#    kappa2_ = kappas[1]     # concentration for the 2nd von Mises member 
#    kappa3_ = kappas[2]     # concentration for the 3rd von Mises member 
#    loc1_ = locs[0]         # location for the 1st von Mises member 
#    loc2_ = locs[1]         # location for the 2nd von Mises member 
#    loc3_ = locs[2]         # location for the 3rd von Mises member 
    
    data = np.zeros((sample_size, num_distr))
    data0 = np.zeros((sample_size, num_distr))
    idx = 0
    for mu, kap in zip(locs, kappas):
        temp_ = stats.vonmises_line.rvs( kap, mu, scal_, sample_size )
        data0[:, idx] = temp_
        if mu > 0.0:
            temp_u = temp_[(temp_ >= transfer_)]
            temp_l = temp_[(temp_ < transfer_)]
            temp_mod = np.concatenate((temp_u - 2.*transfer_, temp_l),axis=0)
        elif mu < 0.0:
            temp_l = temp_[(temp_ <= -transfer_)]
            temp_u = temp_[(temp_ > -transfer_)]
            temp_mod = np.concatenate((temp_u, temp_l + 2.*transfer_),axis=0)
        else:
            temp_mod = temp_
        data[:, idx] = temp_mod
        idx += 1
    
    if len(kappas) < len(ps):
        data[:, idx] = stats.uniform.rvs( loc=u1_, scale=u2_, size=sample_size)
        data0[:, idx] = data[:, idx]
    
    random_idx = np.random.choice( np.arange(num_distr), \
                              size=(sample_size,), p=coefficients )
    X_samples = data[ np.arange(sample_size), random_idx ]
    fig, ax = plt.subplots(1, 1, figsize=(9,3))
    ax.set_title('Random sample from mixture of von Mises and Uniform distributions (ST)')
    ax.hist( X_samples, bins=100, density=True, label='sample-mod', color = 'skyblue' );
    X_samples0 = data0[ np.arange(sample_size), random_idx ]
    ax.hist( X_samples0, bins=100, density=True, label='sample-original', color = 'orange', alpha=0.3 );
    ax.legend()
    
    fig, ax = plt.subplots(1, 1, figsize=(9,3))
    ax.hist( X_samples, bins=100, density=True, label='sample-mod', color = 'skyblue' );
    ax.set_title('Random sample from mixture of von Mises and Uniform distributions (ST)')
    ax.legend()
    
    return X_samples



 
def rs_mix_vonMises2( kappas, locs, ps, sample_size=None ):
    """
    Generate random sample from a mixture of von Mises and/or Uniform 
        distributions, given their parameters. 
    Inspired by the web post: 
        "Creating a mixture of probability distributions for sampling"
            a question on stackoverflow:
                https://stackoverflow.com/questions/47759577/
                creating-a-mixture-of-probability-distributions-for-sampling
    """
    # to change the scale of the von Mises distributions: 
    scal_ = 0.5
    # to account for a uniform distribution: 
    u1_ = -np.pi/2
    u2_ =  np.pi
    kappa1_ = kappas[0]     # concentration for the 1st von Mises member 
    kappa2_ = kappas[1]     # concentration for the 2nd von Mises member 
    kappa3_ = kappas[2]     # concentration for the 3rd von Mises member 
    loc1_ = locs[0]         # location for the 1st von Mises member 
    loc2_ = locs[1]         # location for the 2nd von Mises member 
    loc3_ = locs[2]         # location for the 3rd von Mises member 
    distributions = [
            { "type": stats.vonmises.rvs, \
                 "args": {"kappa":kappa1_, "loc":loc1_, "scale":scal_ }},
            { "type": stats.vonmises.rvs, \
                  "args": {"kappa":kappa2_, "loc":loc2_, "scale":scal_ }},
            { "type": stats.vonmises.rvs, \
                   "args": {"kappa":kappa3_, "loc":loc3_, "scale":scal_ }},
            #{ "type": stats.uniform.rvs,  \
            #        "args": {"loc":u1_, "scale":u2_ } }
            ]
    # coefficients = np.array( [ p1_, p2_, p3_ ] ) # these are the weights 
    coefficients = ps
    coefficients /= coefficients.sum() # in case these did not add up to 1
    if sample_size == None:
        sample_size = 1000
        
    transfer_ = np.pi*scal_
    num_distr = len(distributions)
    data = np.zeros((sample_size, num_distr))
    datab = np.zeros((sample_size, num_distr))
    for idx, distr in enumerate(distributions):
        temp_ = distr["type"]( **distr["args"], size=(sample_size,))
        datab[:, idx] = temp_
        if locs[idx] > 0.0:
        # if max(temp_) > transfer_:
            temp_u = temp_[(temp_ >= transfer_)]
            temp_l = temp_[(temp_ < transfer_)]
            temp_mod = np.concatenate((temp_u - 2.*transfer_, temp_l),axis=0)
        elif locs[idx] < 0.0:
        # elif min(temp_) < -transfer_:
            temp_l = temp_[(temp_ <= -transfer_)]
            temp_u = temp_[(temp_ > -transfer_)]
            temp_mod = np.concatenate((temp_u, temp_l + 2.*transfer_),axis=0)
        else:
            temp_mod = temp_
        data[:, idx] = temp_mod
        # data[:, idx] = distr["type"]( **distr["args"], size=(sample_size,))
    random_idx = np.random.choice( np.arange(num_distr), \
                                  size=(sample_size,), p=coefficients )
    X_samples0 = data[ np.arange(sample_size), random_idx ]
    fig, ax = plt.subplots(1, 1, figsize=(9,3))
    ax.hist( X_samples0, bins=100, density=True, label='sample-mod', color = 'skyblue' );
    ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
    X_samples0_b = datab[ np.arange(sample_size), random_idx ]
    ax.hist( X_samples0_b, bins=100, density=True, label='sample-original', color = 'orange', alpha=0.3 );
    ax.legend()
    
    X_samples = X_samples0
    fig, ax = plt.subplots(1, 1, figsize=(9,3))
    ax.hist( X_samples, bins=100, density=True, label='sample', color = 'skyblue' );
    ax.set_title('Random sample from mixture of von Mises and Uniform distributions')
    ax.legend()
    
    return X_samples
    

def plot_rv_distribution(X, axes=None):
    """
    Plot the PDF or PMF, CDF, SF, and PPF of a given random variable
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
    x_min_999, x_max_999 = X.interval(0.999)
    x999 = np.linspace(x_min_999, x_max_999, 1000)
    x_min_95, x_max_95 = X.interval(0.95)
    x95 = np.linspace(x_min_95, x_max_95, 1000)
    
    if hasattr(X.dist, 'pdf'):
        axes[0].plot(x999, X.pdf(x999), label='PDF')
        axes[0].fill_between(x95, X.pdf(x95), alpha=0.25)
    else:
        # discrete random variables do not have a pdf method, instead we 
        # use pmf:
        x999_int = np.unique(x999.astype(int))
        axes[0].bar(x999_int, X.pmf(x999_int), label='PMF')
        
    axes[1].plot(x999, X.cdf(x999), label='CDF')
    axes[1].plot(x999, X.sf(x999), label='SF')
    axes[2].plot(x999, X.ppf(x999), label='PPF')
    
    for ax in axes:
        ax.legend()
            
     
        
        
        
        
def plot_mixs_vonMises_Specific(mixX, angles):
    """
    Plot the PDF , CDF, SF, and PPF of a random variable that follows the von 
    Mises distribution
    """
    n_clus = mixX.n_clusters
    
    x_min_, x_max_ = np.radians((min(angles), max(angles)))
    x999 = np.linspace(x_min_, x_max_, 1000)
    tXpdf = np.zeros(len(x999),)
    tXcdf = np.zeros(len(x999),)
    tXsf = np.zeros(len(x999),)
    tXppf = np.zeros(len(x999),)
    
    fig, axes = plt.subplots(n_clus+1, 4, figsize=(15, 3*(n_clus+1)))
    for i in range(n_clus):
        str_1 = 'von Mises for X_' + str(i+1)
        # location in rads: 
        loc = math.atan2( mixX.cluster_centers_[i,1], \
                             mixX.cluster_centers_[i,0])
        print('loc=',loc)
        # concentration: 
        kappa = mixX.concentrations_[i]
        # weight: 
        weight = mixX.weights_[i]
        # construct the member von Mises:
        X = stats.vonmises( kappa, loc )
        # the mixture of the von Mises individuals: 
        tXpdf += weight*X.pdf(x999)
        tXcdf += weight*X.cdf(x999)
        tXsf += weight*X.sf(x999)
        tXppf += weight*X.ppf(x999)
        # the confidence interval 95(%): 
        x_min_95, x_max_95 = vonmises.interval(0.90, kappa, loc)
        print('a=',x_min_95)
        print('b=',x_max_95)
        # x_min_95, x_max_95 = X.interval(0.95)
        x95 = np.linspace(x_min_95, x_max_95, 1000)
        # construct the PDF:
        # axes[i, 0].plot(x999, vonmises.pdf(x999, kappa, loc), label='PDF')
        axes[i, 0].plot(x999, X.pdf(x999), label='PDF')
        axes[i, 0].set_ylabel(str_1)
        axes[i, 0].fill_between(x95, X.pdf(x95), alpha=0.25)
        axes[i, 1].plot(x999, X.cdf(x999), label='CDF')
        axes[i, 1].plot(x999, X.sf(x999), label='SF')
        axes[i, 2].plot(x999, X.ppf(x999), label='PPF')
        stats.probplot(stats.vonmises.rvs(kappa, loc, size=len(x999)), \
                       dist=stats.vonmises, \
                       sparams=(kappa, loc), plot=axes[i, 3])
        axes[i, 0].legend()
        axes[i, 1].legend()
        axes[i, 2].legend()
        
    axes[n_clus, 0].plot(x999, tXpdf, label='PDF')
    axes[n_clus, 1].plot(x999, tXcdf, label='CDF')
    axes[n_clus, 1].plot(x999, tXsf, label='SF')
    axes[n_clus, 2].plot(x999, tXppf, label='PPF')
    axes[n_clus, 0].set_ylabel('Mixture von Mises')
    # rr = stats.vonmises.rvs(kappa, loc, size=len(x999))
    # axes[n_clus, 3].plot(x999, rr, label='random PDF')
    # axes[n_clus, 3].hist(rr, normed=True, histtype='stepfilled', alpha=0.2)
    axes[n_clus, 0].legend()
    axes[n_clus, 1].legend()
    axes[n_clus, 2].legend()


def plot_mixs_vonMises_General(mixX):
    """
    Plot the PDF , CDF, SF, and PPF of a random variable that follows the von 
    Mises distribution
    """
    n_clus = mixX.n_clusters
    
    if n_clus != 1:
        fig, axes = plt.subplots(n_clus, 3, figsize=(12, 9))
        
    for i in range(n_clus):
        # location in rads: 
        loc = math.atan2( mixX.cluster_centers_[i,1], \
                             mixX.cluster_centers_[i,0])
        # concentration: 
        kappa = mixX.concentrations_[i]
        # weight: 
        # weight = mixX.weights_[i]
        # construct the member von Mises:
        X = stats.vonmises( kappa, loc )
        # plot the PDF, CDF, SF and PPF, 
        # by calling the function 'plot_rv_distribution': 
        if n_clus == 1:
            plot_rv_distribution(X, axes=None)
        else:
            plot_rv_distribution(X, axes=axes[i, :])
#        # alternatively, create the plot herein: 
#        x_min_999, x_max_999 = X.interval(0.999)
#        # print('a=',x_min_999)
#        # print('b=',x_max_999)
#        x999 = np.linspace(x_min_999, x_max_999, 1000)
#        x_min_95, x_max_95 = X.interval(0.95)
#        x95 = np.linspace(x_min_95, x_max_95, 1000)
#        # construct and plot the PDF:
#        # axes[i, 0].plot(x999, vonmises.pdf(x999, kappa, loc), label='PDF')
#        axes[i, 0].plot(x999, X.pdf(x999), label='PDF')
#        # shade the 95% interval of the PDF: 
#        axes[i, 0].fill_between(x95, X.pdf(x95), alpha=0.25)
#        # construct and plot the CDF:
#        axes[i, 1].plot(x999, X.cdf(x999), label='CDF')
#        # construct and plot the SF:
#        axes[i, 1].plot(x999, X.sf(x999), label='SF')
#        # construct and plot the PPF:
#        axes[i, 2].plot(x999, X.ppf(x999), label='PPF')
        

def plot_dist_samples(X, X_samples, title=None, ax=None):
    """
    Plot the PDF and histogram of samples of a continuous random variable
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
    x_lim = X.interval(.99)
    x = np.linspace(*x_lim, num=100)
    
    ax.plot(x, X.pdf(x), label='PDF', lw=3)
    ax.hist(X_samples, label='samples', normed=1, bins=75)
    ax.set_xlim(*x_lim)
    ax.legend()
    
    if title:
        ax.set_title(title)
        
    return ax


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:27:24 2018
Author Dimitrios Fafalis 

Import this package as: DFmle

Contents of the package:
        
    Log-likelihood functions of mixture distributions: 
    --------------------------------------------------
    1. logLik_1vM
    2. logLik_2vM
    3. logLik_3vM
    4. logLik_2vM1U
    5. logLik_1vM1U

@author: df
"""

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------- #
def logLik_1vM( theta, *args ):
    """
    The negative log-likelihood function 'l' to be minimized
    in order to estimate the von Mises distribution parameters:
        kappa1: the concentration of the von Mises distribution
        mu1: the location of the von Mises distribution
        theta:= [ kappa1, mu1 ]
        args[0]:= X_samples: the observations sample 
        args[1]:= I_samples: the intensities of the observations sample 
    The function returns the negative log-likelihood function. 
    This function is to be called with optimize.minimize function of scipy:
        results = optimize.minimize( logLik_2vM, in_guess, args=(r_X,Int) ...)
    """
    scal_ = 0.5
    x_ = np.array(args[0]).T
    i_ = np.array(args[1]).T
    
    # the unknown parameters:
    kappa = theta[0]
    mu = theta[1]
    
    # the 1st von Mises distribution on semi-circle:
    fvm_ = stats.vonmises.pdf( x_, kappa, mu, scal_ )
    
    # Calculate the negative log-likelihood as the negative sum of the log 
    # of the PDF of a von Mises distributions, with location mu and 
    # concentration kappa: 
    logLik = -np.sum( np.multiply( np.log( fvm_ ), i_ ) )
        
    return logLik


# ---------------------------------------------------------------------- #
def logLik_2vM( theta, *args ):
    """
    The negative log-likelihood function 'l' to be minimized
    in order to estimate the mixture parameters:
        p1, p2: the weights of the 2 von Mises distributions on semi-circle
        kappa1, kappa2: the concentrations of the 2 von Mises distributions
        mu1, mu2: the locations of the 2 von Mises distributions
        theta:= [ p1, kappa1, mu1, p2, kappa2, mu2 ]
        args[0]:= X_samples: the observations sample 
        args[1]:= I_samples: the intensities of the observations sample 
    The function returns the negative log-likelihood function. 
    This function is to be called with optimize.minimize function of scipy:
        results = optimize.minimize( logLik_2vM, in_guess, args=(r_X,Int) ...)
    """
    # the scale is 1.0 for the circle and 0.5 for the semi-circle: 
    scal_ = 0.5
    x_ = np.array(args[0]).T
    i_ = np.array(args[1]).T
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    p2_ = theta[3]
    kappa2_ = theta[4]
    m2_ = theta[5]
    
    # the 1st von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    
    # the 2nd von Mises distribution on semi-circle:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
        
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_
    
    logMix = -np.sum( np.multiply( np.log( fm_ ), i_ ) )
    
    return logMix


# ---------------------------------------------------------------------- #
def logLik_3vM( theta, *args ):
    """
    The negative log-likelihood function 'l' to be minimized
    in order to estimate the mixture parameters:
        p1, p2, p3: the weights of the 3 von Mises distributions on semi-circle
        kappa1, kappa2, kappa3: the concentrations of the 3 von Mises 
        mu1, mu2, mu3: the locations of the 3 von Mises distributions
        theta:= [ p1, kappa1, mu1, p2, kappa2, mu2, p3, kappa3, mu3 ]
        args[0]:= X_samples: the observations sample 
        args[1]:= I_samples: the intensities of the observations sample 
    The function returns the negative log-likelihood function. 
    This function is to be called with optimize.minimize function of scipy:
        results = optimize.minimize( logLik_3vM, in_guess, args=(r_X,Int) ...)
    """
    # the scale is 1.0 for the circle and 0.5 for the semi-circle: 
    scal_ = 0.5
    x_ = np.array(args[0]).T
    i_ = np.array(args[1]).T
    # print(type(x_)), print('x_=', x_.shape)
    # print(type(i_)), print('i_=', i_.shape)
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    p2_ = theta[3]
    kappa2_ = theta[4]
    m2_ = theta[5]
    p3_ = theta[6]
    kappa3_ = theta[7]
    m3_ = theta[8]
    
    # the 1st von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    # logvM1 = -np.sum( stats.vonmises.logpdf( x_, kappa1_, m1_, scal_ ) )
    
    # the 2nd von Mises distribution on semi-circle:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
    # logvM2 = -np.sum( stats.vonmises.logpdf( x_, kappa2_, m2_, scal_ ) )
        
    # the 3rd von Mises distribution on semi-circle:
    fvm3_ = stats.vonmises.pdf( x_, kappa3_, m3_, scal_ )
    
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ + p3_*fvm3_
    # logMix = logvM1 + logvM2 + logU - len(x_)*(np.log(p1_*p2_*pu_))
    
    # if the r.s. is the x_ only: 
    # logMix = -np.sum( np.log( fm_ ) )
    # if the r.s. is equally-spaced angles with varying intensity i_: 
    logMix = -np.sum( np.multiply( np.log( fm_ ), i_ ) )
    
    return logMix


# ---------------------------------------------------------------------- #
def logLik_2vM1U( theta, *args ):
    """
    The negative log-likelihood function 'l' to be minimized
    in order to estimate the mixture parameters:
        p1, p2: the weights of the 2 von Mises distributions on semi-circle
            pu: the weight of the uniform distribution on the semi-circle
        kappa1, kappa2: the concentrations of the 2 von Mises distributions
        mu1, mu2: the locations of the 2 von Mises distributions
        theta:= [ p1, kappa1, mu1, p2, kappa2, mu2, pu ]
        args[0]:= X_samples: the observations sample 
        args[1]:= I_samples: the intensities of the observations sample 
    The function returns a vector F with the derivatives of 'l' wrt the 
        components of theta. 
    The function returns the negative log-likelihood function. 
    This function is to be called with optimize.minimize function of scipy:
        results = optimize.minimize( logLik_2vM1U, in_guess, args=(r_X,Int) ..)
    """
    scal_ = 0.5
    x_ = np.array(args[0]).T
    i_ = np.array(args[1]).T
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    p2_ = theta[3]
    kappa2_ = theta[4]
    m2_ = theta[5]
    pu_ = theta[6]
    
    # the 1st von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    
    # the 2nd von Mises distribution on semi-circle:
    fvm2_ = stats.vonmises.pdf( x_, kappa2_, m2_, scal_ )
    
    # the uniform distribution:
    xu_1 = min(x_)
    xu_2 = (max(x_) - min(x_))
    fu_ = stats.uniform.pdf( x_, loc=xu_1, scale=xu_2 )
    
    # mixture distribution: 
    fm_ = p1_*fvm1_ + p2_*fvm2_ + pu_*fu_
    
    logMix = -np.sum( np.multiply( np.log( fm_ ), i_ ) )
    
    return logMix


# ---------------------------------------------------------------------- #
def logLik_1vM1U( theta, *args ):
    """
    The negative log-likelihood function 'l' to be minimized
    in order to estimate the mixture parameters:
        p1:     the weight of the von Mises distribution on semi-circle
        pu:     the weight of the uniform distribution on the semi-circle
        kappa1: the concentration of the von Mises distribution
        mu1:    the location of the von Mises distributions
        theta:= [ p1, kappa1, mu1, pu ]
        args[0]:= X_samples: the observations sample (the angles in our case)
        args[1]:= I_samples: the intensities of the observations sample 
    The function returns a vector F with the derivatives of 'l' wrt the 
        components of theta. 
    The function returns the negative log-likelihood function. 
    This function is to be called with optimize.minimize function of scipy:
        results = optimize.minimize( logLik_1vM1U, in_guess, args=(r_X,Int) ..)
    """
    scal_ = 0.5
    x_ = np.array(args[0]).T
    i_ = np.array(args[1]).T
    
    # the unknown parameters:
    p1_ = theta[0]
    kappa1_ = theta[1]
    m1_ = theta[2]
    pu_ = theta[3]
    
    # the von Mises distribution on semi-circle:
    fvm1_ = stats.vonmises.pdf( x_, kappa1_, m1_, scal_ )
    
    # the uniform distribution:
    xu_1 = min(x_)
    xu_2 = (max(x_) - min(x_))
    fu_ = stats.uniform.pdf( x_, loc=xu_1, scale=xu_2 )
    
    # mixture distribution: 
    fm_ = p1_*fvm1_ + pu_*fu_
    
    logMix = -np.sum( np.multiply( np.log( fm_ ), i_ ) )
    
    return logMix


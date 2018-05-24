#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:36:15 2018

determine the fibers orientation using MA image processing 

@author: df
"""

# packages needed to run MA modules
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# import numpy as np
import os
import MA.ImageProcessing as MAIP
import MA.OrientationAnalysis as MAOA
# import MA.Tools as MATL
 
# 
import dispersionCalculator as dC

# ------------------------------------------------------------------------- #
# define the path where the images are imported from:
# im_dir = '/Users/df/0_myDATA/testSamples/'
# im_path = im_dir + 'MAX_20X_Airyscan_6.jpg'
# OutputRoot = 

# the directory where the images-to-process are kept: 
im_dir = '/Users/df/Documents/0-DATA/s1_20180223/MAX_20180223_S1_20X_1c/'
# the image to process: 
im_name = 'MAX_20180223_S1_20X_1c.png'
im_name_0 = os.path.splitext(im_name)[0]
# the complete path to the image file
im_path = im_dir + im_name
# the root of the path to save the results: 
# ( it is extended by names added from the functions to follow )
OutputRoot = im_dir + im_name_0
# if I want one more folder inside:
# OutputRoot_ = im_dir + im_name_0 + '/' + im_name_0

# transform the image to a numpy array:
img = MAIP.GetImageMatrix(im_path)

# create an object from the OrientationAnalysis class:
OAnalysis = MAOA.OrientationAnalysis(
            BaseAngFolder = "/Users/df/AngleFiles",
            OutputRoot = OutputRoot,
            )
# analyze the image:
OAnalysis.SetImage(img)

# generate the results:
ResultsFFT = OAnalysis.ApplyFFT()
ResultsFFT.PlotHistogram()
ResultsGrad = OAnalysis.ApplyGradient()
ResultsGrad.PlotHistogram()

# ResultsFFT.X -> the angles of the histogram
# ResultsFFT.Y -> the intensity of the FFT for the previous angles

# ------------------------------------------------------------------------- # 
# now call DF functions which call the spherecluster module to fit the 
# intensity histogram to a mixture of von-Mises-Fischer distribution
# angles = ResultsFFT.X
# values = ResultsFFT.Y
# number of clusters:
n_clust = 2
res_vonMF = dC.data2vonMises( ResultsFFT.X, ResultsFFT.Y, n_clust, OutputRoot )

# check goodness-of-fitness measure R2:
fhat, Fhat, Fexp = dC.myCDFvonMises( ResultsFFT.X, ResultsFFT.Y, res_vonMF )
R2 = dC.myR2( Fhat, Fexp )
dC.myPPplot( Fhat, Fexp, OutputRoot, n_clust )

# plot the fitted distribution and vectors on the image:
n_X = dC.normalizeIntensity( ResultsFFT.X, ResultsFFT.Y )
dC.plotMixvMises1X2( res_vonMF, im_path, n_X )


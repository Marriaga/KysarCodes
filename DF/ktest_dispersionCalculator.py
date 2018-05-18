# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:53:34 2018

@author: DF
"""

#import csv
#import numpy as np

import dispersionCalculator as dC

import time

#from PIL import Image
# import glob

import os, os.path

myPath = '/Users/df/0_myDATA/testSamples/'

YNtest = 'No'

if YNtest == 'No':
    print('Real simulation ...')
    
    #im_name = input('Give the name of the image file: ')
    # im_name = "MAX_noepi_C2.png"
    ##im_name = "SR_01.png"
    #csv_name = input('Give the name of the csv file: ')
    # csv_name = "MAX_noepi_C2.csv"
    ##csv_name = "SR_01_FFT.csv"
    
    # if you have a different file path:
    # im_path = myPath + 'MAX_20X_Airyscan_6.jpg'
    # csv_path = myPath + "MAX_20X_Airyscan_6.csv"
    #
    im_path = myPath + 'Clipboard.png'
    csv_path = myPath + "Dir_hist_FFT.csv"
    
    # this is how to split the path from the file name:
    # use it in 
    temp_path, temp_file = os.path.split(im_path)
    print(temp_path)
    print(temp_file)
        
    # number of clusters:
    n_clust = 3
    
    angles, values = dC.imageCVS2Data( csv_path )
    
    # ------------- 
    # if you have the angles and values and n_clust, then type the following
    # lines (between the dash-lines) in your code:
    tt1 = time.process_time()
    res_vonMF = dC.data2vonMises( angles, values, n_clust, im_path )
    tt2 = time.process_time()
    total_time = tt2 - tt1
    
    # check goodness-of-fitness measure R2:
    fhat, Fhat, Fexp = dC.myCDFvonMises( angles, values, res_vonMF );
    R2 = dC.myR2( Fhat, Fexp )
    
    dC.myPPplot( Fhat, Fexp, im_path, n_clust )
    
    print(total_time)
    print(res_vonMF)
    print('R^2 =',R2)
    
    YNplot = 'Yes'
    if YNplot == 'Yes':
        
        n_X = dC.normalizeIntensity( angles, values )
        #n_X_new = n_X.copy()
        #n_X_new[:,1] = n_X_new[:,1] - min(n_X_new[:,1])
        dC.plotMixvMises1X2( res_vonMF, im_path, n_X )
        
    # ------------- 
    print('... finished!')

else:
    print('Test simulation ... ')
    ress = dC.test()
    print('... finished!')


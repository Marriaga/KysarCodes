# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 19:02:36 2017
Author Dimitrios Fafalis 

Contents of the package:
    
    Import this package as: DFDC or dC

Package sklearn is required to be installed 

@author: DF
"""

# Dimi2 module
import math
import numpy as np
import os, os.path

from matplotlib import pyplot as plt

from scipy import stats
# from scipy.stats import vonmises

# Import package spherecluster for fitting mixtures of von Mises:
# from spherecluster import VonMisesFisherMixture
# instead, import the following modified version which reads directly the 
# intensity of image's FFT:
from sphereclusterIntensity import VonMisesFisherMixtureInt

def __init__():
    pass

def imageCVS2Data( csv_file ):
    """
    Given the csv file 'csv_file' created by the Directionality module of 
    the ImageJ package for an image file 'im_name', return the angles and 
    the intensity fields as separate arrays, to be used to identify the
    parameters of the mixture of von Mises distributions describing the fibers
    orientation
    """
    #csv_file = input('Give the name of the csv file: ')
    # or use numpy function loadtxt as follows: 
    mydata = np.loadtxt(csv_file, skiprows=1, delimiter=',')
    angles = mydata[:,0]
    values = mydata[:,1]
#    # =========================== 
#    # or use csv package as follows: 
#    Index = []; # for the ref number(not in use for us)
#    Angles = []; # the bins in angles (degrees) for which the intensity 
#                 # is measured in col-3
#    Vals = []; # the measured ligth intensity, col-3
#
#    with open(csv_file) as csvDataFile:
#        csvReader = csv.reader(csvDataFile)
#        for row in csvReader:
#            Index.append(row[0])
#            Angles.append(row[1])
#            Vals.append(row[2])
#
#    # the angles (in degrees):
#    anglesAr = np.array(Angles[1:])
#    angles = anglesAr.astype(np.float)
#    # the intensity (-):
#    valsaAr = np.array(Vals[1:])
#    values = valsaAr.astype(np.float)
#    
#    print(values - myvals)
#    print(angles - myangles)
#    # =========================== 
    
    return angles, values, mydata

def data2vonMises( angles, values, n_clust=1, im_path=None ):
    """
    Provide as input the angles and values (intensity) from the FFT of image
    'im_path', and the number of clusters n_clust
    to get the estimated parameters of the von Mises mixture
    
    Parameters:
    -----------
    angles : array-like, shape=(bin_angle, ), (usually 2 degrees width)
    values : array-like, shape=(bin_angle, ), intensity of light per bin angle
    n_clust : integer, the number of von Mises distributions of the mixture
    im_path : string, the name of the image file from which the fibers 
                directions are extracted
                
    Returns:
    --------
    res_vonMF : array-like, shape=(n_clust,4)
    col-1 : direction in rads
    col-2 : direction in degrees
    col-3 : concentration (dispersion)
    col-4 : weight of member von Mises distribution
    """
    
    # create an object from the sphereclusterIntensity module:
    vmf_soft = VonMisesFisherMixtureInt( n_clusters=n_clust, \
                                        posterior_type='soft', n_init=20 )
    
#    # ============== 
#    # test-1: 
#    n_X = normalizeIntensity( angles, values )
#    
#    p_X = makePoints( n_X )
#    
#    X = getCosSin( p_X )
#    
#    Int = np.ones_like( p_X )
#    
#    vmf_soft.fit(X,Int=None)
#    # ============== 
    
    # ============== 
    # test-2: 
    # convert the angles from degrees to radians:
    r_X = np.radians( angles )
    # compute the cosines and sines of the radiant angles r_X:
    # X is the input to the fit function of spherecluster module.
    X = getCosSin( r_X )
    n_X = normalizeIntensity( angles, values )
    Int = n_X[:,1]
    # Int = n_X[:,1] - min(n_X[:,1])
    # ============== 
    
    # call function fit of sphereclusterIntensity module:
    vmf_soft.fit( X, Int )
    
    # save the fit results into an array:
    res_vonMF = saveFitMixvMises( vmf_soft, im_path )
    
    return res_vonMF, vmf_soft

def normalizeIntensity( angles, values, YNplot='Yes' ):
    """Function to normalize the light intensity data from FFT.
    Parameters:
    -----------
    angles : array-like, shape=(bin_angle, ), (usually 2 degrees width)
    values : array-like, shape=(bin_angle, ), intensity of light per bin angle
    
    Returns:
    --------
    n_X   : array, shape=(bin_angle, 2)
    col-1 : bins of angles in degrees, same as in angles
    col-2 : normalized intensity of light per bin angle: [angles, n_values]
    """
    # the x-axis along which to integrate: angles 
    # the values of the function to integrate: values 
    
    # number of equally-spaced segments in [-pi/2, pi/2]
    bin_angle = len(angles)
    n_X = np.zeros((bin_angle,2))
    n_X[:,0] = angles
    
    y_intQ = np.trapz(values, x=np.radians(angles))
    # the normalized data:
    n_values = values/y_intQ
    n_X[:,1] = n_values
    
    if YNplot == 'Yes':
        xlim_min = min(angles)
        xlim_max = max(angles)
        # plot the normalized data:
        fig, ax = plt.subplots()
        ax.plot(angles, n_values, 'r')
        ax.set_title('Normalized intensity data')
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_xlabel('degrees')
        ax.set_ylabel('Intensity from FFT')
        ax.legend()
    
    return n_X

def makePoints( n_X, YNplot='Yes' ):
    """Function to convert the normalized intensity into populations per angle
    Parameters:
    -----------
    n_X : array-like, shape=(bin_angle, 2)
    col-1 : bins of angles in degrees (usually 2 degrees width)
    col-2 : normalized intensity of light per bin angle: [angles, n_values]

    Returns:
    --------
    p_X : array-like, shape=(n_pop, )
    col-1 : angles in rads of n_X[:,0]
    """
    # the equally-spaced segments in the space [-pi/2, pi/2]:
    bin_angle, _ = np.shape(n_X)
    # population factor:
    sfact = 3.0
    # convert the intensity to points per angle:
    #p_Xi = np.round( sfact*n_X[:,1] / max(n_X[:,1]) )
    #p_Xi = np.round( sfact*(n_X[:,1]/min(n_X[:,1])) )
    p_Xi = np.ceil( sfact*(n_X[:,1]/min(n_X[:,1])) )
    # convert them to integers type:
    p_Xi = p_Xi.astype(int)
    # the total population for all angles:
    #p_Xi_sum = sum(p_Xi)
    p_Xd = np.zeros((0,))
    for i in range(bin_angle):
        Xtemp = np.repeat(n_X[i,0], p_Xi[i])
        p_Xd = np.append(p_Xd, Xtemp)
    # convert the degrees to radians:
    p_X = np.radians(p_Xd)
    print(sum(p_Xi))
    
    if YNplot == 'Yes':
        # plot the XX array into a histogram and the X onto the unit circle:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].set_title('Histogram of points from intensity')
        axes[0].hist( p_X, bins=bin_angle, label='radians', alpha=0.5 );
        axes[0].set_xlabel( 'radians' )
        axes[0].legend()
        axes[1].plot( p_X, 'b.', markersize=1, label='data populated', alpha=0.5 );
        axes[1].set_ylabel( 'rads' )
        axes[1].legend()
        plt.show()

    return p_X
     
def getCosSin( p_X, YNplot='Yes' ):
    """Function to compute the cosine and sine of a vector array in radians
    Parameters:
    -----------
    p_X : array-like, shape=(n_pop,)
    col-1 : angles in rads 

    Returns:
    --------
    Xsc : array-like, shape=(n_pop,2)
    col-1 : cosine of p_X
    col-2 : sine of p_X
    """

    Xsc = np.zeros((len(p_X), 2))
    Xsc[:,0] = np.cos(p_X)
    Xsc[:,1] = np.sin(p_X)
    
    if YNplot == 'Yes':
        # plot the data on the unit circle:
        fig, ax = plt.subplots()
        ax.plot( Xsc[:,0], Xsc[:,1], 'r.' );
        ax.set_title('The angles distributed on the unit circle');
        ax.set_xlim( -1.0, 1.0 );
        ax.set_ylim( -1.0, 1.0 );
        ax.axis( 'equal' );

    return Xsc
    
def saveFitMixvMises( mixvM, im_path ):
    """Function to save the fitting parameters of von Mises mixture 
    into a .txt file and also produce and save a figure
    Parameters:
    -----------
    mixvM : the object created by class VonMisesFisherMixture
    im_name : the image file (usually .png) 
                from which the fiber mapping is identified
    Returns:
    --------
    res_vonMF : array-like, shape=(n_clus,4)
    col-1 : locations of clusters in rads
    col-2 : locations of clusters in degrees
    col-3 : concentrations of clusters (-)
    col-4 : weights of clusters (-)
    """
    n_clus = mixvM.n_clusters

    res_vonMF = np.zeros((n_clus,4))
    for i in range(n_clus):
        temp = math.atan2( mixvM.cluster_centers_[i,1], \
                          mixvM.cluster_centers_[i,0])
        res_vonMF[i,:] = [ temp, np.degrees(temp), \
                             mixvM.concentrations_[i], mixvM.weights_[i] ]

    # create a name for the output file:
    # first split the path from the file name:
    temp_path, im_name = os.path.split(im_path)
    #print(temp_path)
    #print(temp_file)
    arr_name = 'Res_' + im_name.replace(".png","") + '_' \
                + str(n_clus) + 'vM' + '.csv'

    # save the .txt file:
    arr_name_path = temp_path + '/' + arr_name
    np.savetxt( arr_name_path, res_vonMF, fmt='%.18e', delimiter=', ', \
                newline='\n', header='location_rad, location_deg, concentration,weight', footer='', comments='# ')

    print(res_vonMF)
    return res_vonMF
    
def plotMixvMises1X2( res_vonMF, im_path, n_X ):
    """
    To plot the fitted mixture of von Mises distributions
    Parameters:
    -----------
    res_vonMF : array-like, shape=(n_clust,4), the results stored
    im_path   : string, the name of the image file
    n_X       : the normalized intensity
    
    Output:
    -----------
    fig_name : an image .png file
    """
    from PIL import Image, ImageDraw
    #import os.path
    
    img_0 = Image.open(im_path)
    draw_0 = ImageDraw.Draw(img_0)
    
    temp_path, im_name = os.path.split(im_path)
    im_name_0 = os.path.splitext(im_name)[0]
    draw_0.text((-1,0), text=im_name, fill=100)
    im_cen = img_0.size
    pYmax = im_cen[1]
    #cen_x = int(min(im_cen)/2)
    #cen_y = int(max(im_cen)/2)
    cen_x = int(im_cen[0]/2)
    cen_y = int(im_cen[1]/2)

    angles = n_X[:,0]
    nvalls = n_X[:,1]
    
    xlim_min = min(angles)
    xlim_max = max(angles)
    rlim_min = np.radians(xlim_min)
    rlim_max = np.radians(xlim_max)
    
    n_clus = len(res_vonMF)
    # plot the distributions:
    # x = np.linspace(-np.pi/2, np.pi/2, num=100)
    x = np.linspace(rlim_min, rlim_max, num=100)
    xdegs = np.degrees(x)
    fX_tot = np.zeros(len(x),)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    #axes[2].imshow(np.asarray(img_0), cmap='gray');
    for i in range(n_clus):

        temp_r = res_vonMF[i,0]; # location in rads 
        temp_d = np.round(res_vonMF[i,1], decimals=2); # location in degrees 
        temp_c = res_vonMF[i,2]; # concentration 
        temp_w = res_vonMF[i,3]; # weights

        str_1 = 'von Mises for X_' + str(i+1)
        fX_i = stats.vonmises( temp_c, temp_r )
        #fX_i = stats.vonmises(res_vonMF[i,2], res_vonMF[i,0])
        axes[0].plot( xdegs, fX_i.pdf(x), label=str_1 );

        # str_2 = 'weighted von Mises for X_' + str(i+1)
        str_2 = 'von Mises X_' + str(i+1)
        axes[1].plot( xdegs, temp_w*fX_i.pdf(x), '--', label=str_2 );
        # sum individual distributions weighted to get total mixture von Mises:
        fX_tot += temp_w*fX_i.pdf(x)
        # annotate as text the locations of the individual von Mises:
        axes[1].annotate( temp_d, xy=(temp_d, temp_w*fX_i.pdf(temp_r)), \
            xytext=(temp_d, temp_w*fX_i.pdf(temp_r)), \
            arrowprops=dict(facecolor='black', shrink=0.5), )
        
        #tmpThe = temp_r - np.pi/2
        start_x = cen_x*(1 - np.cos(temp_r))
        start_y = cen_y*(1 - np.sin(temp_r))
        start_y = pYmax - start_y
        end_x = cen_x*(1 + np.cos(temp_r))
        end_y = cen_y*(1 + np.sin(temp_r))
        end_y = pYmax - end_y
        draw_0.line( [(start_x,start_y),(end_x,end_y)], fill=i+1+i*50, width=2 )
        axes[2].annotate( temp_d, xy=(end_x,end_y), xytext=(end_x,end_y), \
            color='g')

    axes[1].plot( xdegs, fX_tot, 'r', label="combined von Mises" );
    axes[1].plot( angles, nvalls, 'k--', label="normalized data" );
    axes[1].set_title(im_name)
    axes[1].set_xlabel('degrees')
    axes[1].set_ylabel('pdf')
    # axes[1].legend(loc=1)
    # axes[1].legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=3)
    axes[1].legend(bbox_to_anchor=(0.75, 0.99),loc=2, borderaxespad=0., ncol=2)
    axes[0].legend()

    axes[2].imshow(np.asarray(img_0), cmap='gray');

    # create a name for the output image file:
    #fig_name = 'Res_' + im_name.replace(".png","") + '_' + str(n_clus) + 'vM'
    fig_name = 'Res_' + im_name_0 + '_' + str(n_clus) + 'vM' + '_fit'
    fig_name_path = temp_path + '/' + fig_name
    # save the .png and .eps files:
    fig.savefig(fig_name_path + '.png')
    fig.savefig(fig_name_path + '.eps')

    return
            
def myR2( Fhat, Fexp ):
    """
    Function to compute the coefficient of determination R2, 
    a measure of quality of fit. 
    """
    
    Fbar = np.mean(Fhat)
    
    FexpmFhat = sum((Fexp - Fhat)**2)
    
    num = sum((Fhat - Fbar)**2)
    
    den = num + FexpmFhat
    
    R2 = num / den
        
    return R2

def myCDFvonMises( angles, values, res_vonMF ):
    """
    res_vonMF : array-like, shape=(n_clust,4)
        col-1 : direction in rads
        col-2 : direction in degrees
        col-3 : concentration (dispersion)
        col-4 : weight of member von Mises distribution
    """      
    x = np.radians(angles)
    p = np.linspace(0.0, 1.0, num=angles.size)
    fX_tot = np.zeros(len(x),)
    cfX_tot = np.zeros(len(x),)
    pfX_tot = np.zeros(len(x),)
    dx = np.radians(abs(angles[0] - angles[1]))

    n_X = normalizeIntensity( angles, values )
    cfX_exp = np.cumsum(n_X[:,1]*dx)

    n_clust = len(res_vonMF)
    for i in range(n_clust):
        
        temp_r = res_vonMF[i,0]; # location in rads 
        # temp_d = np.round(res_vonMF[i,1], decimals=2); # location in degrees 
        temp_c = res_vonMF[i,2]; # concentration 
        temp_w = res_vonMF[i,3]; # weights
        
        fX_i = stats.vonmises( temp_c, temp_r )
        
        # sum individual distributions weighted to get total mixture von Mises:
        fX_tot += temp_w*fX_i.pdf(x)
        # retrieve the cdf: 
        cfX_tot += temp_w*fX_i.cdf(x)
        # retrieve the ppf:
        pfX_tot += temp_w*fX_i.ppf(p)
        
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.set_title('cdf and pdf')
    ax.plot(x, np.cumsum(fX_tot*dx),
      'r', lw=2, alpha=0.6, label='total cdf')
    ax.plot(x, cfX_tot,
      'b--', lw=2, alpha=0.6, label='vonmises cdf')
    ax.plot(x, cfX_exp,
      'k', lw=3, alpha=0.6, label='real cdf')
    ax.plot(x, fX_tot,
      'm-', lw=2, alpha=0.6, label='vonmises pdf')
    ax.legend()
#    ax[0].plot(x, np.cumsum(res_vonMF[0,3]*vonmises.pdf(x, res_vonMF[0,2]) * dx),
#      'b:', lw=2, alpha=0.6, label='cdf-1')
#    ax[0].plot(x, np.cumsum(res_vonMF[1,3]*vonmises.pdf(x, res_vonMF[1,2]) * dx),
#      'g:', lw=2, alpha=0.6, label='cdf-2')
#    ax[0].plot(x, np.cumsum( (res_vonMF[0,3]*vonmises.pdf(x, res_vonMF[0,2]) \
#      + res_vonMF[1,3]*vonmises.pdf(x, res_vonMF[1,2])) * dx), 
#      'm--', lw=2, alpha=0.6, label='sum of cdf1+cdf2')
    
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].set_title('cdf of von Mises')
    ax[0].set_ylabel('Cumulative distribution function')
    ax[0].set_xlabel(r'$\theta$ (degrees)')
    ax[0].plot(angles, cfX_tot,
             'k', lw=3, alpha=0.6, label='total cdf')
    ax[1].set_title('Inverse cdf (ppf--percent point function) of von Mises')
    ax[1].set_ylabel('Quantile (degrees)')
    ax[1].set_xlabel('Cumulative probability')
    ax[1].plot(cfX_tot, angles,
      'k', lw=3, alpha=0.6, label='inverse cdf')
    ax[2].plot(p, pfX_tot,
      'k', lw=3, alpha=0.6, label='total ppf')
    #stats.probplot(cfX_tot, plot=ax[2])
    
    for i in ax:
        i.legend()
    
    return fX_tot, cfX_tot, cfX_exp
    
def myPPplot( cfX_tot, cfX_exp, im_path, n_clus ):
    """
    function to plot the P-P probability plot
    for the goodness-of-fit of a distribution 
    """  
    x = np.linspace(0,1,len(cfX_exp))
    
    R2 = myR2( cfX_tot, cfX_exp )
    str_1 = r'$R^2 = $' + str(R2)
    
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(cfX_exp, cfX_tot,
      'b.', lw=2, alpha=0.6, label='P-P')
    ax.plot(x, x,
      'r--', lw=2, alpha=0.6, label='1:1')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Observed Probability')
    ax.set_ylabel('mvMF Probability')
    ax.legend()
    plt.tight_layout()
    plt.text(0.05, 0.8,str_1, ha='left', va='bottom', transform=ax.transAxes)
    
    temp_path, im_name = os.path.split(im_path)
    im_name_0 = os.path.splitext(im_name)[0]
    # create a name for the output image file:
    # fig_name = 'Res_' + im_name.replace(".png","") + '_' + str(n_clus) + 'vM'
    # fig_name = 'Res_' + im_name.replace(".jpg","") + '_' + str(n_clus) + 'vM'
    fig_name = 'Res_' + im_name_0 + '_' + str(n_clus) + 'vM_PP_R2'
    fig_name_path = temp_path + '/' + fig_name
    # save the .png and .eps files:
    fig.savefig(fig_name_path + '.png')
    fig.savefig(fig_name_path + '.eps')
    
    return

def circ_measures(angles, values):
    """
    Function to compute basic measures for circular statistics
    """
    # convert the angles from degrees to radians:
    r_X = np.radians( angles )
    # compute the cosines and sines of the radiant angles r_X:
    # X is the input to the fit function of spherecluster module.
    CS = getCosSin( r_X )
    C = np.dot(values,CS[:,0])
    print('C=',C)
    S = np.dot(values,CS[:,1])
    print('S=',S)
    R = np.sqrt(C**2 + S**2)
    print('R=',R)
    # since we work with FFT intensities, divide by sum(values) and not 
    # by np.size(angles)
    Rb = R/sum(values)
    print('Rb=',Rb)
    alpha0 = np.arctan(S/C)
    print('alpha0=',alpha0)
    alpha0d = np.degrees(alpha0)
    print('alpha0d=',alpha0d)
    
    return C, S, R, Rb, alpha0, alpha0d
    
def test():
    myPath = '/Users/df/myFiberDirectionality/testSamples/'
    im_path = myPath + 'MAX_20X_Airyscan_6.jpg'
    csv_path = myPath + 'MAX_20X_Airyscan_6.csv'
    n_clust = 3
    angles, values = imageCVS2Data( csv_path )
    res_vonMF = data2vonMises( angles, values, n_clust, im_path )
    YNplot = 'Yes'
    if YNplot == 'Yes':
        n_X = normalizeIntensity( angles, values )
        plotMixvMises1X2( res_vonMF, im_path, n_X )

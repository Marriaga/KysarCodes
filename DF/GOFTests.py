#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:02:13 2018
Updated on Tue Nov 13 --:--:-- 2018

Author Dimitrios Fafalis: Post-Doctoral Researcher at Dr. Kysar's Lab

Contents of the package:
    
    Import this package as:
        
        import DF.GOFTests as DFGOF (or GOF)
    
    The package contains one class: 
        
        "class GOFTests()"
        
    How to use this class: 
    
    Create an instance of this class as follows: 
        
        1. in the initializer of Fitting() class:
            
            self.GOFTests = DF.GOFTests()
            
            self.gofresults = None
        
        2. in the getGOF(self) instance function of the Fitting() class: 
            
            def getGOF(self):
                self.GOFTests.UpdateData( self.Angles, self.Intensities, 
                                     self.parameters, self.N_VonMises, self.Uniform )
                self.gofresults = self.GOFTests.computeGOF()
                self.computeGOF()
                return self.gofresults
                
        3. the information needed as input for class GOFTests comes from 
            the results of the object related to the Fitting() class of MA.
                        
    The class "GOFTests" contains the following instance methods: 
    
    A. specific statistical methods (pre/post-processing of data):
    --------------------------------------------------------------  
    1. Get_PDF_VMU
    2. Get_CDF_VMU
    3. makePoints
    4. CDFX
    5. ECDF
    
    B. Methods for Goodness-of-fit tests: 
    -------------------------------------
    1. Watson_CDF
    2. Watson_GOF
    3. Kuiper_CDF
    4. Kuiper_GOF
    5. myR2
    6. R2_GOF
    7. Plot_PP_GOF
    8. Uniformity_test
    
    C. Other functions: 
    -------------------
    1. UpdateData
    2. computeGOF
    3. dispersion_coeff

Contents (by order of occurance): 
    1.  __init__
    2.  UpdateData
    3.  computeGOF
    4.  Watson_GOF
    5.  Watson_CDF
    6.  Kuiper_GOF
    7.  Kuiper_CDF
    8.  CDFX
    9.  Get_PDF_VMU
    10. Get_CDF_VMU
    11. makePoints
    12. Uniformity_test
    13. ECDF
    14. myR2
    15. R2_GOF
    16. Plot_PP_GOF
    17. dispersion_coeff

@author: df
"""

# import python packages to use: 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
#from scipy import stats
#from scipy.stats import vonmises
import pandas as pd


class GOFTests():
    '''
    How to call this class from another class: 
        self.GOFTests.UpdateData( self.Angles, self.Intensities, 
                                  self.parameters, self.N_VonMises, self.Uniform )
    '''
    
    # ---------------------------------------------------------------------- #
    # Initializer / Instance Attributes: 
    # 1. new initializer: 
    def __init__(self, Angles=None, Intensities=None, parameters=None, N_VonMises=None, Uniform=None):
        self.Angles = Angles
        self.Intensities = Intensities
        self.parameters = parameters
        self.N_VonMises = N_VonMises
        self.Uniform = Uniform
        self.gofresults = None
    
    # 2. update the results: 
    def UpdateData( self, Angles=None, Intensities=None, parameters=None, N_VonMises=None, Uniform=None):
        if Angles is not None: self.Angles = Angles
        if Intensities is not None: self.Intensities = Intensities
        if parameters is not None: self.parameters = parameters
        if N_VonMises is not None: self.N_VonMises = N_VonMises
        if Uniform is not None: self.Uniform = Uniform
        self.gofresults = None

    # ---------------------------------------------------------------------- #
    # 3. Compute and Collect GOF results
    def computeGOF(self):
        """
        Compute and Collect results from the following GOF tests: 
            1. Watson
            2. Kuiper
            3. R2
        Returns:
        --------
            self.gofresults
        """
        # Watson GOF test: 
        watson_data = self.Watson_GOF( alphal=2 )
        
        # Kuiper GOF test: 
        kuiper_data = self.Kuiper_GOF( )
        
        # R2 coefficient of determination: 
        r2_data = self.R2_GOF( )
        
        # if you want to keep only the common columns from the data frames
        #   of the tests, use the following names_ list: 
        names_ = ['GOFTest', 'Symbol', 'Statistic', 'CriticalValue','PValue', \
                  'SignifLevel', 'Decision']
        
        self.gofresults = pd.concat([watson_data, kuiper_data, r2_data])
        self.gofresults = self.gofresults[names_]
           
    # ---------------------------------------------------------------------- #
    # 4. the Watson GOF test: 
    def Watson_GOF( self, alphal=2 ):
        """
        From Book: "Applied Statistics: Using SPSS, STATISTICA, MATLAB and R"
                    by J. P. Marques de Sa, Second Edition
                    SPringer, ISBN: 978-3-540-71971-7 
                    Chapter 10-Directional Statistics, Section 10.4.3 
        Watson U2 test of circular data X.
        Parameters:
        -----------
            self.CDFX(): the EXPECTED distribution (cdf) from the fitting.
        Evaluates:
        ----------
            U2: the Watson's test statistic
            US: the modified test statistic for known (loc, kappa)
            UC: the critical value at ALPHAL (n=100 if n>100)
            where:
            ------
            ALPHAL: 1->0.1; 2->0.05 (default); 3->0.025; 4->0.01; 5->0.005
        Returns:
        --------
            Watson_gof_results: dataFrame collecting the above.
        References: 
        ----------
            1. Equations (6.3.34), (6.3.36) of:
                Mardia K. V., Jupp P. E., (2000)
                "Directional Statistics", Wiley.
                ISBN: 0-471-95333-4 
            2. Equation (7.2.8) of:
                Jammalamadaka S. Rao, SenGupta A. (2001), 
                "Topics in Circular Statistics", World Scientific.
                ISBN: 981-02-3778-2 
        """
        
        V = self.CDFX()
        n = len(V)
        Vb = np.mean(V)
        cc = np.arange(1., 2*n, 2)
        
        # The Watson's statistic: 
        # Eq. (6.3.34) of Ref. [1]: 
        U2 = np.dot(V, V) - np.dot(cc, V)/n + n*(1./3. - (Vb - 0.5)**2.)
        
        # The modified Watson's statistic (when both loc and kappa are known): 
        # Eq. (6.3.36) of Ref. [1]: 
        Us = (U2 - 0.1/n + 0.1/n**2)*(1.0 + 0.8/n)
        
        alpha_lev = np.array([0.1, 0.05, 0.025, 0.01, 0.005])
        #              alpha=   =0.1     =0.05   =0.025  =0.01   =0.005
        c = np.array([ [ 2,     0.143,	0.155,	0.161,	0.164,	0.165 ],
                       [ 3,     0.145,	0.173,	0.194,	0.213,	0.224 ],
                       [ 4,     0.146,	0.176,	0.202,	0.233,	0.252 ],
                       [ 5,     0.148,	0.177,	0.205,	0.238,	0.262 ],
                       [ 6,     0.149,	0.179,	0.208,	0.243,	0.269 ],
                       [ 7,     0.149,	0.180,  0.210,	0.247,	0.274 ],
                       [ 8, 	    0.150,	0.181,	0.211,	0.250,	0.278 ],
                       [ 9,     0.150,	0.182,	0.212,	0.252,	0.281 ],
                       [ 10,    0.150,	0.182,	0.213,	0.254,	0.283 ],
                       [ 12,	    0.150,	0.183,	0.215,	0.256,	0.287 ],
                       [ 14,    	0.151,	0.184,	0.216,	0.258,	0.290 ],
                       [ 16,    	0.151,	0.184,	0.216,	0.259,	0.291 ],
                       [ 18,	    0.151,	0.184,	0.217,	0.259,	0.292 ],
                       [ 20,    	0.151,	0.185,	0.217,	0.261,	0.293 ],
                       [ 30,	    0.152,	0.185,	0.219,	0.263,	0.296 ],
                       [ 40,    	0.152,	0.186,	0.219,	0.264,	0.298 ],
                       [ 50,	    0.152,	0.186,	0.220,	0.265,	0.299 ],
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
        pval2 = self.Watson_CDF( U2 )
        pvals = self.Watson_CDF( Us )
        
        if U2 < uc:
            H0_W = "Do not reject"
        else:
            H0_W = "Reject"
        
        Watson_gof_results = pd.DataFrame({'GOFTest': "Waston", \
                                           'Symbol': "U2", \
                                           'Statistic': U2, \
                                           'StatisticStar': Us, \
                                           'CriticalValue': uc, \
                                           'PValue': pval2, \
                                           'ModifiedPValue': pvals, \
                                           'SignifLevel': alpha_lev[alphal], \
                                           'Decision': [H0_W]})
        
        Watson_gof_results = Watson_gof_results[['GOFTest', 'Statistic', \
                                                 'Symbol', \
                                                 'StatisticStar', \
                                                 'CriticalValue','PValue', \
                                                 'ModifiedPValue', 'SignifLevel', \
                                                 'Decision']]
    
        return Watson_gof_results
        # return U2, Us, uc, pval2, pvals, H0_W, alpha_lev[alphal]
      
    # ---------------------------------------------------------------------- #
    # 5. The Watson CDF function: 
    def Watson_CDF( self, x ):
        """
        Function to return the value of the CDF of the asymptotic Watson 
        distribution at x: 
        Parameters:
        -----------
            x: (scalar) value to compute the above CDF.
        Returns:
        --------
            y: (scalar) the CDF of asymptotic Watson computed at x.
        References: 
        ----------
            1. Equation (6.3.37) of:
                Mardia K. V., Jupp P. E., (2000)
                "Directional Statistics", Wiley.
                ISBN: 0-471-95333-4 
            2. Equation (7.2.10) of:
                Jammalamadaka S. Rao, SenGupta A. (2001), 
                "Topics in Circular Statistics", World Scientific.
                ISBN: 981-02-3778-2 
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
            
        #print('Watson k iterations =',k)
        return y
        
    # ---------------------------------------------------------------------- #
    # 6. the Kuiper GOF test: 
    def Kuiper_GOF( self ):
        """
        cY_:  the probability distribution of the postulated model on ordered data
        Kuiper Vn test of circular data X.
        Parameters:
        -----------
            self.CDFX(): the EXPECTED distribution (cdf) from the fitting.
        Evaluates:
        ----------
            Vn: the Kuiper's test statistic
            pVn: the p-value for the statistic
            Vns: the modified Kuiper's test statistic
            pVns: the p-value for the modified statistic
            Vc: the critical value at ALPHAL (n=100 if n>100)
            alp_lev: the alpha level of significance
            where:
            ------
            ALPHAL: 1->0.1; 2->0.05 (default); 3->0.025; 4->0.01 
        Returns:
        --------
            Kuiper_gof_results: dataFrame collecting the above.
        References: 
        ----------
            1. Equation (6.3.25) of:
                Mardia K. V., Jupp P. E., (2000)
                "Directional Statistics", Wiley.
                ISBN: 0-471-95333-4 
            2. Equation (7.2.2) of:
                Jammalamadaka S. Rao, SenGupta A. (2001), 
                "Topics in Circular Statistics", World Scientific.
                ISBN: 981-02-3778-2 
        """
        cY_ = self.CDFX()
        n = len(cY_)
        sqn = np.sqrt(n)
        
        # prepare the Kuiper test: 
        ii = np.arange(0, 1, 1/n)
        vec1 = cY_ - ii
        jj = np.arange(1/n, 1+1/n, 1/n)
        vec2 = jj - cY_
        
        Dp = max(vec1)
        Dm = max(vec2)
        
        # compute the Kuiper statistic Vn: 
        Vn = sqn*(Dp + Dm)
        # compute the Kuiper p-value: 
        pVn = self.Kuiper_CDF( Vn, n )
        
        # modified Kuiper statistic Vns: 
        # eq. (6.3.30) without the sqrt(n): 
        # for n>=8, both Vn and Vns are very close. 
        Vns = Vn * (1 + 0.155/sqn + 0.24/n)
        pVns = self.Kuiper_CDF( Vns, n )
        
        # Significance levels: 
        alpha = np.array([ 0.10, 0.05, 0.025, 0.01 ])
        # The Vc provided below are the upper quantiles of Vns,
        # Table 6.3 of Ref. [1].
        Vc = np.array([ 1.620, 1.747, 1.862, 2.001 ])
        
        if Vns < min(Vc):
            H0_K = "Do not reject"
            dif_v = abs(Vn - Vc)
            ind_v = np.argmin(dif_v)
            #print('ind_v:',ind_v)
            alp_lev = alpha[ind_v]
            Vcc = Vc[ind_v]
        else:
            H0_K = "Reject"
            Vcc = Vc[-1]
            alp_lev = 1
        
        Kuiper_gof_results = pd.DataFrame({'GOFTest': "Kuiper", \
                                           'Statistic': Vn, \
                                           'Symbol': "Vn", \
                                           'ModStatistic': Vns, \
                                           'CriticalValue': Vcc, \
                                           'PValue': pVn, \
                                           'ModifiedPValue': pVns, \
                                           'SignifLevel': alp_lev, \
                                           'Decision': [H0_K]})
        
        Kuiper_gof_results = Kuiper_gof_results[['GOFTest', 'Statistic', \
                                                 'Symbol', 'ModStatistic', \
                                                 'CriticalValue','PValue', \
                                                 'ModifiedPValue', \
                                                 'SignifLevel', \
                                                 'Decision']]
        
        #return Vn, Vcc, pVn, H0_K, alp_lev
        return Kuiper_gof_results    

    # ---------------------------------------------------------------------- #
    # 7. the Kuiper CDF function: 
    def Kuiper_CDF( self, x, n ):
        """
        Function to return the value of the CDF of the asymptotic Kuiper
        distribution at x: 
        Parameters:
        -----------
            x: (scalar) value to compute the above CDF.
            n: (integer) the number of points available from the observed data
        Returns:
        --------
            y: (scalar) the CDF of asymptotic Kuiper computed at x.
        References: 
        ----------
            1. Equation (6.3.29) of:
                Mardia K. V., Jupp P. E., (2000)
                "Directional Statistics", Wiley.
                ISBN: 0-471-95333-4 
            2. Equation (7.2.6) of:
                Jammalamadaka S. Rao, SenGupta A. (2001), 
                "Topics in Circular Statistics", World Scientific.
                ISBN: 981-02-3778-2 
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
            
        #print('Kuiper k iterations =', k)
        return y
    
    # ---------------------------------------------------------------------- #
    # 8. the CDF for the Fitted Distribution: 
    def CDFX( self ):
        """
        Function to get the CDF of FITTED or EXPECTED distribution 
            over the observed x-axis points.
        Called in methods:
            'Watson_GOF' 
            'Kuiper_GOF' 
        Parameters:
        -----------
            self.makePoints(): returns the observed points over which the CDFX
                                will be evaluated.
        Returns:
        --------
            cdfX: the requested CDF
        """
        p_X = self.makePoints()
        aa = np.sort( p_X )
        
        cX_t = self.Get_CDF_VMU( aa )
                
        if max(cX_t) > 1:
            cdfX = cX_t - (max(cX_t) - 1.)
        else:
            cdfX = cX_t
        
        return cdfX
    
    # ---------------------------------------------------------------------- #
    # 9. get the PDF for the mixture of VMU model: 
    def Get_PDF_VMU( self, x ):
        """
        Parameters:
        -----------
            x: any vector over which to compute the PDF of the mixture VMU 
        
        Returns:
        --------
            pdf_vmu: the PDF for the underlying mixture of Von Mises and 
                     Uniform distributions, calculated over the vector x.
        """
        pdf_vmu = np.zeros_like(self.Angles)
        p_, kappa_, m_, pu_ = self.parameters
        #print(p_, kappa_, m_, pu_)
        for vmi in range( self.N_VonMises ): # p, k, m 
            pdf_vmu += p_[vmi] * sp.stats.vonmises.pdf( x, kappa_[vmi], \
                                                 loc = m_[vmi], scale = 0.5 )
        
        if self.Uniform:
            pdf_vmu += pu_ * sp.stats.vonmises.pdf( x, 1.0E-3, \
                                                    loc = 0.0, scale = 0.5 )
        
        return pdf_vmu
        
    # ---------------------------------------------------------------------- #
    # 10. get the CDF for the mixture of VMU model: 
    def Get_CDF_VMU( self, x ):
        """
        Parameters:
        -----------
            x: any vector over which to compute the CDF of the mixture VMU 
        
        Returns:
        --------
            cdf_vmu: the CDF for the underlying mixture of Von Mises and 
                     Uniform distributions, calculated over the vector x.
        """
        cdf_vmu = np.zeros_like(x)
        p_, kappa_, m_, pu_ = self.parameters
        for vmi in range( self.N_VonMises ): # p, k, m 
            cdf_vmu += p_[vmi] * sp.stats.vonmises.cdf( x, kappa_[vmi], \
                                                 loc = m_[vmi], scale = 0.5 )
           
        if self.Uniform:
            cdf_vmu += pu_ * sp.stats.vonmises.cdf( x, 1.0E-3, \
                                                    loc = 0.0, scale = 0.5 )
        
        return cdf_vmu
    
    # ---------------------------------------------------------------------- #
    # 11. generate points on the semi-circle based on the intensities: 
    def makePoints( self ):
        """
        Function to convert the normalized intensities into points per angle
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
        Int = self.Intensities
        Angs = np.degrees(self.Angles)
        
        # the equally-spaced segments in the space [-pi/2, pi/2]:
        bin_angle = len( Int )
        
        # population factor:
        sfact = 3.0
        
        # convert the intensity to points per angle:
        #p_Xi = np.round( sfact*n_X[:,1] / max(n_X[:,1]) )
        #p_Xi = np.round( sfact*(n_X[:,1]/min(n_X[:,1])) )
        p_Xi = np.ceil( sfact*(Int/min(Int)) )
        
        # convert them to integer type:
        p_Xi = p_Xi.astype(int)
        
        # the total population for all angles:
        #p_Xi_sum = sum(p_Xi)
        
        # make the points: 
        p_Xd = np.zeros((0,))
        for i in range(bin_angle):
            Xtemp = np.repeat(Angs[i], p_Xi[i])
            p_Xd = np.append(p_Xd, Xtemp)
        
        # convert degrees to radians:
        p_X = np.radians(p_Xd)
            
        return p_X
    
    # ---------------------------------------------------------------------- #
    # 12. test for the isotropicity (uniformity) of angles: 
    def Uniformity_test( self ):
        '''
        Assesement of Uniformity of Raw data:
            using a uniform probability plot:
                plot the sorted observations theta_i/pi against i(n+1)
                If the data come from a uniform distribution, then the points
                should lie near a straight line of unit slope passing through 
                the origin.
            The function also returns the R2 "coefficient of determination" 
                for that fit: the closer to unity the more uniform the data.
        '''
        # get the observed data: 
        pX = ( self.makePoints() + np.pi/2 )/np.pi
        n = len(pX)
        h = np.arange(1/(n+1), n/(n+1), 1/(n+1))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(h, pX[1::], color='r', linestyle=':', label='data')
        #ax.plot([0, 1], [0, 1], color='g', linestyle='--', label='45$^o$ line')
        ax.plot(h, h, color='g', linestyle='--', label='45$^o$ line')
        ax.square()
        ax.legend()
        
        # compute the R2 coefficient of determination: 
        SSE = sum((h - pX[1::])**2)
        SSTO = sum((h - np.mean(h))**2)
        R2 = 1 - SSE/SSTO
        ax.annotate('$R^2=%s$' % R2, xy=(0.4,1.0), xytext=(0.4, 1.0) )
        
        return R2
    
    # ---------------------------------------------------------------------- #
    # 13. the CDF for the Experimental Data: 
    def ECDF( self, data ):
        """
        CAUTION!: NOT for Light Intensity (from FFT) data! 
        Function to evaluate the Empirical Distribution Function (ECDF) 
        associated with the empirical measure of a sample. 
        Parameters:
        -----------
            data: random sample, (rads) in the nature of the problem to consider 
        Returns:
        --------
            x_values:   the x-axis of the ECDF figure (rads)
            y_values:   the y-axis of the ECDF figure, the ECDF
        called in method: 
            'R2_GOF' 
        """
        # put the data into an array:
        raw_data = np.array(data)
        # create a sorted series of unique data:
        cdfx = np.sort(np.unique(raw_data))
        # raw_data = data
        # # create a sorted series of unique data
        # cdfx = np.sort(data)
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
    
    # ---------------------------------------------------------------------- #
    # 14. coefficient of Determination R2 for the fitted distribution: 
    def myR2( self, Fexp, Fobs ):
        """
        Function to compute the coefficient of determination R2, 
        a measure of goodness-of-fit.
        Parameters:
        -----------
            Fobs: empirical (observed) CDF,
                    can be computed from the function: 
                        ECDF_Intensity( angles, values )
            Fexp: expected (postulated) CDF
        Returns:
        --------
            R2 
        """
        
        Fbar = np.mean(Fexp)
        
        FobsmFexp = sum((Fobs - Fexp)**2)
        
        num = sum((Fexp - Fbar)**2)
        
        den = num + FobsmFexp
        
        R2 = num / den
                
        return R2
    
    # ---------------------------------------------------------------------- #
    # 15. GOF based on the value of R2: 
    def R2_GOF( self ):
        """
        Function to prepare the data for the calculation of the R2 coefficient.
        Returns:
        --------
            R2_results: dataFrame collecting R2 results.
        """
        # get the observed data: 
        pX = self.makePoints()
            #print('size pX: ', np.size(pX))
            #print('pX1: ', pX[0:6])
        # compute the empirical cumulative distribution function: 
        x_obs, cdf_obs = self.ECDF( pX )        
        
        # compute the expexted probability distribution function for the fit:
        fX_r2 = self.Get_PDF_VMU( x_obs )
        
        # compute the expexted cumulative distribution function for the fit:
        dx = np.diff(x_obs)
        cdfX_r2 = np.ones(len(x_obs),)
        cdfX_r2[0:-1] = np.cumsum(fX_r2[0:-1]*dx)
        
        # compute the R2 coefficient: 
        R2 = self.myR2( cdfX_r2, cdf_obs )
        
        # this is subjective: what should a good R2 be???
        # print('R2 =', R2)
        R2c = 0.90
        if R2 > R2c:
            H0_R2 = "Do not reject"
        else:
            H0_R2 = "Reject"
        
        R2_results = pd.DataFrame({'GOFTest': "R2", 'Statistic': R2, \
                                   'Symbol': "R2", 'CriticalValue': R2c, \
                                   'Decision': [H0_R2]})
        
        R2_results = R2_results[['GOFTest', 'Statistic', 'Symbol', \
                                 'CriticalValue', 'Decision']]
        
        # plot the PP-plot: 
        PPfig = self.Plot_PP_GOF( cdfX_r2, cdf_obs )
        
        return R2_results
        #return R2, H0_R2
    
    # ---------------------------------------------------------------------- #
    # 16. Generate the Probability-Probability plot: 
    def Plot_PP_GOF( self, Fexp, Fobs ):
        """
        Generate the Probability-Probability plot.
        Parameters:
        -----------
            Fexp: distribution function of postulated (expected) distribution
            Fobs: empirical distribution function (the original data)
        """
        x = np.linspace(0,1,len(Fobs))
        
        x = np.linspace(0,1,len(Fobs))
        
        fig, ax = plt.subplots(1,1,figsize=(4,3))
        ax.plot(Fexp, Fobs, 'k.', lw=2, alpha=0.6, label='P-P plot')
        ax.plot(x, x, 'r--', lw=2, alpha=0.6, label='1:1')
        ax.set_title('P-P plot for a Mixture Model')
        ax.set_xlabel('Mixture distribution function')
        ax.set_ylabel('Empirical distribution function')
        ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
        ax.legend()
        
        return fig

    # ---------------------------------------------------------------------- #
    # 17. compute the dispersion coefficient: 
    def dispersion_coeff( self ):
        """
        Return the dispersion coefficient of one family of dispersed fibers,
        as defined by Holpzapfel.
        kappa_ip = ( 1 - I1(b) / I0(b) )/2      (eq. 2.23)
        Ref: Holzapfel et al. (2015), "Modelling non-symmetric collagen fibre 
            dispersion in arterial walls", J. R. Soc. Interface 12: 20150188 
        """
        # Collect the dispersion coefficients for every fiber family: 
        dispersion = []
        p_, kappa_, m_, pu_ = self.parameters
        for vmi in range( self.N_VonMises ): # p, k, m 
            kap_ = kappa_[vmi]
            dispersion.append( 0.5*(1 - (sp.special.i1(kap_))/(sp.special.i0(kap_)) ) )
        
        return dispersion
    
# End of Class GOFTests 
# -------------------------------------------------------------------------- #       
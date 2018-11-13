#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import str
from builtins import range
from builtins import object
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import progressbar
import os

import MA.ImageProcessing as MAIP
import MA.Tools as MATL

import DF

from joblib import Parallel, delayed
import multiprocessing

import pandas as pd

#Get gauss distribution value     
def Gauss(x,s=1,D=1,ori=0.0):
    if s==1 and D==1:
        amp=0.5641895835477562869480794515607725858
    else:
        amp=1.0*D/(s*np.sqrt(np.pi))
    return amp*np.exp(-(x-ori)**2/(s**2))
    
#Sames as Gauss but for default PoU gauss
def QuickGauss(x):
    return 0.5641895835477562869480794515607725858*np.exp(-(x**2))

#Adjusted Gauss for Angles    
def GetGauss(Mang,ang,**kwargs):
    if kwargs is None:
        return QuickGauss(np.minimum(np.abs(Mang-ang),np.abs(np.abs(Mang-ang)-180)))
    else:
        return Gauss(np.minimum(np.abs(Mang-ang),np.abs(np.abs(Mang-ang)-180)),**kwargs)




def FormatShape(Shape,ErrVarId="Shape"):
    # Get Image Shape
    if type(Shape) == type(int(0)):
        NewShape=(Shape,Shape)
    elif type(Shape) == type((0,0)):
        NewShape = Shape
    elif type(Shape) == type(np.array([0])):
        NewShape = Shape.shape
    else:
        print("Value of '" +ErrVarId+ "' not valid")
    return NewShape


class ImageWindow(object):
    def __init__(self,Shape=512,WType="Tukey",Alpha=-1,PType="Radial",PadShape=-1):
        # Shape - Shape of Image
        # S -> SxS image
        # (M,N) -> MxN image
        # Array -> size(array) image
        
        # WType - String with type of Windowing (as given by WTypes_Dict)
        # PType - String with partition type (Radial or Separable)
        # Alpha - Float with value of alpha (-1 -> use filter default)
        # PadShape - Pad Image with zeros so that it ends up with this shape

        # Dictionary of Implemented Windowing WTypes:
        # Name -> (Function , Default Alpha)
        self.WTypes_Dict = dict([
        ("None",(None,None)),
        ("Hann",(self.Hann1D,0.5)),
        ("Hamming",(self.Hamming1D,0.54)),
        ("Blackman",(self.Blackman1D,0.16)),
        ("Tukey",(self.Tukey1D,0.50))
        ])
        
        # Initialize
        self.Shape = None
        self.WTypeName = None
        self.WTypeFunc = None
        self.Alpha = None
        self.PType = None
        self.WindowArray = None
        self.PadShape=None
        
        self.SetShape(Shape)
        self.SetWType(WType)
        self.SetAlpha(Alpha)
        self.SetPType(PType)
        self.SetPadShape(PadShape)
        
    def SetShape(self,Shape):
        NewShape=FormatShape(Shape)
            
        if self.Shape != NewShape:
            self.Shape = NewShape
            self.WindowArray = None
            
    def SetWType(self,WType):
        NewWTypeName = None
        # Set Windowing WType
        if type(WType) == type("string"):
            if WType in list(self.WTypes_Dict.keys()):
                NewWTypeName = WType
            else:
                print("Windowing name (" + WType + ")not implemented")
        else:
            print("Windowing type (" + WType + ")not implemented")
            
        if self.WTypeName != NewWTypeName:
            self.WTypeName = NewWTypeName
            self.WTypeFunc = self.WTypes_Dict[NewWTypeName][0]
            self.Alpha = self.WTypes_Dict[NewWTypeName][1]
            self.WindowArray = None

    def SetPType(self,PType):
        NewPType = None
        # Set Partition PType

        if PType in ["Separable","Radial"]:
            NewPType = PType
        else:
            print("Partition name (" + PType + ")not implemented")

        if self.PType != NewPType:
            self.PType = NewPType
            self.WindowArray = None
    
    def SetAlpha(self,Alpha):
        NewAlpha = None
        
        if Alpha == -1:
            NewAlpha = self.WTypes_Dict[self.WTypeName][1]
            
        elif Alpha>=0 and Alpha<=1:
            NewAlpha = Alpha
            
        if self.Alpha != NewAlpha:
            self.Alpha = NewAlpha
            self.WindowArray = None       

    def SetPadShape(self,PadShape):
        # Set Partition PadShape
        if PadShape == -1:
            NewPadShape=None
        else:
            NewPadShape = FormatShape(PadShape,ErrVarId="PadShape")

        if self.PadShape != NewPadShape:
            self.PadShape = NewPadShape
            self.WindowArray = None
    
    def Apply(self,Matrix):
        if self.WTypeName == "None":
            return Matrix
        if self.WindowArray is None:
            self.ComputeWindowArray()
        return Matrix*self.WindowArray
        
    def Reset(self,Shape=None,WType=None,Alpha=None,PType=None,PadShape=None):
        if Shape is not None: self.SetShape(Shape)
        if WType is not None: self.SetWType(WType) 
        if Alpha is not None: self.SetAlpha(Alpha) 
        if PType is not None: self.SetPType(PType) 
        if PadShape is not None: self.SetPadShape(PadShape)
        
    def GetWindowArray(self):
        if self.WTypeName == "None":
            return None
        if self.WindowArray is None:
            self.ComputeWindowArray()
        return self.WindowArray
        
    def ComputeWindowArray(self):
        self.WindowArray = np.fromfunction(self.Builder2D,self.Shape,)

    def Builder2D(self,i,j):
        M ,N  = np.shape(i)
        LM,LN = ((M,N) if self.PadShape is None else self.PadShape)
        zi=(i/(M-1)-1/2)*M/LM
        zj=(j/(N-1)-1/2)*N/LN
        
        if self.PType=="Separable":
            return self.WTypeFunc(zi,self.Alpha)*self.WTypeFunc(zj,self.Alpha)
        elif self.PType == "Radial":
            zr=np.sqrt(zi**2+zj**2)
            return self.WTypeFunc(zr,self.Alpha)
        else:
            raise ValueError("Type must be 'Separable' or 'Radial'")   
        
    ### Windowing Types ###
        
    # Hamming
    def Hamming1D(self,z,alph=0.54):
        z=np.minimum(1.0,np.maximum(0.0,z+1/2))
        return alph-(1-alph)*np.cos(2*np.pi*z)  
        
    # Hann
    def Hann1D(self,z,alph=0.5):
        return self.Hamming1D(z,0.5)
        
    # Blackman    
    def Blackman1D(self,z,alph=0.16):
        z=np.minimum(1.0,np.maximum(0.0,z+1/2))
        # a0=(1-alph)/2
        # a1=1/2
        # a2=alph/2
        # w=a0-a1*np.cos(2*np.pi*z)+a2*np.cos(4*np.pi*z)
        return ((1-alph)-np.cos(2*np.pi*z)+alph*np.cos(4*np.pi*z))/2  
        
    # Tukey    
    def Tukey1D(self,z,alph=0.50):
        z=np.minimum(1.0,np.maximum(0.0,z+1/2))
        z=np.maximum(np.minimum(z,alph/2),z-1+alph)
        z/=alph

        return (1-np.cos(2*np.pi*z))/2  
           
class OrientationAnalysis(object):
    def __init__(self,BaseAngFolder=None,OutputRoot=None,verbose=True):
        self.Window=ImageWindow()
        self.ImageMatrix=None
        self.ImageShape=None
        self.WorkImageMatrix=None
        
        self.OutputRoot = OutputRoot
        self.BaseAngFolder = BaseAngFolder
        
        self.OutputWindowed = MATL.MakeRoot(self.OutputRoot,"_"+self.GetWindowPropName())

        self.FFTAnalysis = FFTAnalysis(BaseAngFolder=BaseAngFolder,verbose=verbose)
        self.GradientAnalysis = GradientAnalysis()
     
    def GetWindowPropName(self):
        return self.Window.WTypeName + "_" + str(self.Window.Alpha) + "_" + str(self.Window.PType)[:3]

    def SetImage(self,Image,AdjustOutputRoot=True):
        NewOutputRoot = None
        if type(Image) == type(" ") and os.path.isfile(Image):
            if AdjustOutputRoot: 
                Fold,File=os.path.split(Image)
                Fold = Fold if self.OutputRoot is None else os.path.dirname(self.OutputRoot)
                ImgName = os.path.splitext(File)[0]
                NewOutputRoot = os.path.join(Fold,ImgName)
            NewImageMatrix = MAIP.GetImageMatrix(Image)
        elif type(Image) == type(np.array([0])):
            NewImageMatrix = Image
        else:
            raise ValueError("Image must be a path to a file or a numpy array")
           
            
        if (self.ImageMatrix != NewImageMatrix).any():
            if NewOutputRoot is not None: self.OutputRoot = NewOutputRoot
            self.ImageMatrix = NewImageMatrix
            self.ImageShape = NewImageMatrix.shape
            self.SetWindowProperties(Shape=self.ImageShape)
            # self.WorkImageMatrix=None

    def SetWindowProperties(self,Shape=None,WType=None,Alpha=None,PType=None,PadShape=None):
        if self.Window is None: self.Window=ImageWindow() 
        self.Window.Reset(Shape=Shape,WType=WType,Alpha=Alpha,PType=PType,PadShape=PadShape)
        self.OutputWindowed = MATL.MakeRoot(self.OutputRoot,"_"+self.GetWindowPropName())
        self.WorkImageMatrix=None
            
    def ApplyWindow(self):
        if self.Window is not None:
            self.WorkImageMatrix = self.Window.Apply(self.ImageMatrix)
        else:
            self.WorkImageMatrix = self.ImageMatrix
        MAIP.SaveShowImage(MAIP.Rescale8bit(self.WorkImageMatrix),self.OutputWindowed)

    def ApplyFFT(self,NewImage=None,**kwargs):
        if NewImage is not None: self.SetImage(NewImage)
        if self.WorkImageMatrix is None: self.ApplyWindow()

        self.FFTAnalysis.OutputRoot=self.OutputWindowed
        ANGS,VALS,NameProp = self.FFTAnalysis.Apply(self.WorkImageMatrix,**kwargs)
        return OrientationResults(ANGS,VALS,OutputRoot=self.OutputWindowed+"_"+NameProp+"_FFT")
        

    def ApplyGradient(self,NewImage=None):
        if NewImage is not None: self.SetImage(NewImage)
        #if self.WorkImageMatrix is None: self.ApplyWindow()
        
        self.GradientAnalysis.OutputRoot=self.OutputRoot
        ANGS,VALS,NameProp = self.GradientAnalysis.Apply(self.ImageMatrix)
        return OrientationResults(ANGS,VALS,OutputRoot=self.OutputRoot+"_"+NameProp+"_Gradient")
        
    
class AngleFilters(object):
    def __init__(self,Size=512,NBins=181,Scale=4,BaseAngFolder=None,FilterRadially=True,verbose=True):
        self.Size = None
        self.NBins = None
        self.Scale = None
        self.BaseAngFolder = None
        self.AngFolder = None
        self.FilterRadially = None
        self.Mang = None
        self.Mrad = None
        self.verbose=None
        
        if BaseAngFolder is None: BaseAngFolder = "temp"
        self.Reset(Size=Size,NBins=NBins,Scale=Scale,BaseAngFolder=BaseAngFolder,FilterRadially=FilterRadially,verbose=verbose)

        self.WarnNoBackup = False

    def Reset(self,Size=None,NBins=None,Scale=None,BaseAngFolder=None,FilterRadially=None,verbose=None):
        if Size is not None: self.Size = Size
        if NBins is not None: self.NBins = NBins
        if Scale is not None: self.Scale = Scale
        if BaseAngFolder is not None: self.BaseAngFolder = BaseAngFolder
        if FilterRadially is not None: self.FilterRadially = FilterRadially
        if verbose is not None: self.verbose = verbose
        self.AngFolder = self.GetAngFolderPath()
        self.Mang = None
        self.Mrad = None
        
    ##==== Files and Folders ====
    # Automatic Folder Name
    def FolderName(self):
        Rad = "Rad" if self.FilterRadially else "Sep"
        return str(self.Size)+"_"+str(self.NBins)+"_"+str(self.Scale)+"_"+Rad
    
    # Automatic Folder Path    
    def GetAngFolderPath(self):
        return os.path.join(self.BaseAngFolder,self.FolderName())
    
    #Get file name for that angle
    def Getfname(self,a):
        return os.path.join(self.AngFolder,str(a)+".npy")

        
    
    ##==== Angles ====
    
    # Compute Matrix with all Angles
    def ComputeMang(self):
        Sa=self.Size*self.Scale
        self.Mang=MAIP.np2Image(np.fromfunction(self._Angles,(Sa,Sa)))

    #Make matrix where each cell is the angle with the center of the figure
    def _Angles(self,i,j):
        S = np.shape(i)[0]
        x=i-S/2
        y=j-S/2
        ang=np.arctan2(y, x)*180.0/np.pi
        negs=ang<0
        ang[negs]=ang[negs]+180
        return ang 
    
    ##==== Radii ====
    
    # Compute Matrix with all Radii
    def ComputeMrad(self):
        Sa=self.Size*self.Scale
        self.Mrad=MAIP.np2Image(np.fromfunction(self._Radii,(Sa,Sa)))
    
    def _Radii(self,i,j):   
        S = np.shape(i)[0]
        x=i-S/2
        y=j-S/2
        rad=np.sqrt(x**2+y**2)
        Grad=rad-S/4 #r_c
        Grad/=S/8 #r_bw
        res=QuickGauss(Grad)
        return res  
    
    ##==== Filters ==== 
   
    # Make folder with rescaled angle matrices for faster processing
    def CreateAngleFiles(self):
        if self.Mang is None: self.ComputeMang()
        MATL.MakeNewDir(self.AngFolder,verbose=self.verbose)
        ANGS=np.linspace(0,180,self.NBins)
        bar = progressbar.ProgressBar(max_value=180.0, term_width=80)
        for a in ANGS:
            bar.update(a)
            fname=self.Getfname(a)
            np.save(fname,self.ComputeAngleFilter(a))
        print(" ")
    
    # Compute the filter for the specified angle
    def ComputeAngleFilter(self,a):
        if self.Mang is None: self.ComputeMang()
        ExpMatrix = GetGauss(self.Mang,a)
        
        if self.FilterRadially:
            if self.Mrad is None: self.ComputeMrad()
            ExpMatrix = ExpMatrix * self.Mrad
        
        return MAIP.ResizeMat(ExpMatrix,self.Size)
    
    # Get the filter for the specified angle from file or computed
    def GetAngleFilter(self,a,Backup=True):
        fname=self.Getfname(a)
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            AngFilter=self.ComputeAngleFilter(a)
            if Backup:
                MATL.MakeNewDir(self.AngFolder,verbose=self.verbose)
                np.save(fname,AngFilter)
            elif not self.WarnNoBackup:
                if self.verbose: print("Warning: Setting Backup to 'False' means that the filter will not be saved.\n \
                       This should be avoided for repetitive runs.")
                self.WarnNoBackup=True
            return AngFilter
            
    def FilterImage(self,Image,Angle,Backup=True):
        AngFilter = self.GetAngleFilter(Angle,Backup=Backup)
        return np.tensordot(AngFilter,Image)
        
class FFTAnalysis(object):
    def __init__(self,BaseAngFolder=None,OutputRoot=None,verbose=True):
        self.AngleFilters=AngleFilters(Size=512,BaseAngFolder=BaseAngFolder,verbose=verbose) #Size=512,NBins=181,Scale=2,BaseAngFolder=None,FilterRadially=True
        self.PowerSpectrum=None
        self.FFTSize = None
        self.verbose=verbose
        self.OutputRoot = OutputRoot
        self.NameProps = self.AngleFilters.FolderName()
    
    def SetAngleFiltersProperties(self,Size=None,NBins=None,Scale=None,BaseAngFolder=None,FilterRadially=None,verbose=None):
        self.AngleFilters.Reset(Size=Size,NBins=NBins,Scale=Scale,BaseAngFolder=BaseAngFolder,FilterRadially=FilterRadially,verbose=verbose)
        self.NameProps = self.AngleFilters.FolderName()

    #Set the center of the Power Spectrum to 0 inside a circle of radius R    
    def MaskPowerSpectrumCenter(self,R):
        def MkCircle(i,j,S,R):
            x=i-S/2
            y=j-S/2
            R2=R**2
            return x**2+y**2<=R2
        self.PowerSpectrum[MAIP.np2Image(np.fromfunction(MkCircle,(self.FFTSize,self.FFTSize),S=self.FFTSize,R=R))]=0
    
    def GetRes(self,Backup=True):
        ## COLLECT WEDGE VALUES
        VALS=np.zeros(self.AngleFilters.NBins)
        ANGS=np.linspace(0,180,self.AngleFilters.NBins)
        
        if self.verbose: bar = progressbar.ProgressBar(max_value=180.0, term_width=80)
        for iangle,Angle in enumerate(ANGS):
            if self.verbose: bar.update(Angle)
            VALS[iangle]=self.AngleFilters.FilterImage(self.PowerSpectrum,Angle,Backup=Backup)
        if self.verbose: print(" ")
        VALS=VALS/np.amax(VALS)
        
        return ANGS,VALS        
        
    def Apply(self,NumpyImage,PSCenter=5,Backup=True):
        self.FFTSize = int(max(NumpyImage.shape)/2)*2 # FFT and Filtering done with even number (not sure if needed)
        self.SetAngleFiltersProperties(Size=self.FFTSize)
        self.PowerSpectrum = self.ComputePowerSpectrum(NumpyImage,self.FFTSize)
        self.MaskPowerSpectrumCenter(PSCenter)
        MAIP.SaveShowImage(self.ScaledPS(),self.OutputRoot,"_PS")

        ANGS,VALS = self.GetRes(Backup=Backup)

        return ANGS,VALS,self.NameProps
        
    #Do FFT, compute power spectrum and return rotated.    
    def ComputePowerSpectrum(self,Matrix,S):
        FFT = np.fft.fftshift(np.fft.fft2(Matrix,s=(S,S)))
        POW = np.real(FFT*np.conj(FFT))
        POW = np.flipud(np.transpose(POW))
        POW[S-int(S/2)-1,int(S/2)]=0 #Remove constant (average pixel intensity)
        return POW      
        
    def ScaledPS(self):
        return MAIP.Rescale8bit(np.log(self.PowerSpectrum+1))
        
class GradientAnalysis(object):
    def __init__(self,OutputRoot=None,NBins=181,SmoothingSteps=10,MovAvgSize=5):
        self.ScaledImage=None
        
        self.NBins = None
        self.SmoothingSteps = None
        self.MovAvgSize = None
        self.SetGradientProperties(NBins=NBins,SmoothingSteps=SmoothingSteps,MovAvgSize=MovAvgSize)
        
        self.OutputRoot = OutputRoot
        self.NameProps = self.GetNameProps()

        
    def SetGradientProperties(self,NBins=None,SmoothingSteps=None,MovAvgSize=None):
        if NBins is not None: self.NBins = NBins
        if SmoothingSteps is not None: self.SmoothingSteps = SmoothingSteps
        if MovAvgSize is not None: self.MovAvgSize = MovAvgSize
        self.NameProps = self.GetNameProps()        
        
    def GetNameProps(self):
        return str(self.NBins)+"_Sm"+str(self.SmoothingSteps)+"_MvA"+str(self.MovAvgSize)
        
        
    def ScaleMat(self,Matrix):
        return (Matrix-np.amin(Matrix))/(np.amax(Matrix)-np.amin(Matrix))
    
    def ComputeMagnitude(self):
        M=self.ScaleMat(np.hypot(self.GradientNP[0],self.GradientNP[1]))
        M[:,0]=M[0,:]=M[:,-1]=M[-1,:]=0 #Fix Boundaries
        return MAIP.np2Image(M)
    
    def ComputeAngles(self):
        AG=np.arctan2(self.GradientNP[1],self.GradientNP[0])*180.0/np.pi
        AG=AG+90.0
        negs=AG<0
        AG[negs]=AG[negs]+180
        toobig=AG>=180
        AG[toobig]=AG[toobig]-180
        return MAIP.np2Image(AG)

    def MakeHist(self):
        AngleBins = np.int32(np.round(self.GradAngles*(self.NBins-1)/180,0))
        ANGS=np.linspace(0,180,self.NBins)
        Y=np.zeros_like(ANGS,dtype=np.float32) 
        ThresholdMagnitude=np.percentile(self.GradMagnitudes,50)
        
        for i in range(self.NBins):
            PixelsWithCorrectAngle= AngleBins==i
            PixelsWithLargeMagnitude= self.GradMagnitudes>ThresholdMagnitude
            RelevantPixels = np.logical_and(PixelsWithCorrectAngle,PixelsWithLargeMagnitude)
            # Y[i]=np.sum(self.GradMagnitudes[RelevantPixels]**2)
            Y[i]=np.sum(RelevantPixels)
            
        Y[0]+=Y[-1]
        VALS=Y
        VALS=MATL.CircularMovingAverage(Y, n=self.MovAvgSize)
        VALS*=1/np.amax(VALS)
        return ANGS,VALS
        
    def Apply(self,NumpyImage):
        self.ScaledImage=self.ScaleMat(NumpyImage)
        self.ScaledImage=MAIP.SmoothImage(self.ScaledImage,N=10)
        self.GradientNP=np.gradient(MAIP.Image2np(self.ScaledImage))
        
        self.GradMagnitudes=self.ComputeMagnitude()
        self.GradAngles=self.ComputeAngles()
        ANGS,VALS = self.MakeHist()

        return ANGS,VALS,self.NameProps    
           
class OrientationResults(object):
    def __init__(self,X,Y,OutputRoot=None):
        self.X=X
        self.Y=Y
        self.OutputRoot=OutputRoot
        
    def GetAI(self):
        '''Wrapper of GetXY for obtaining data to use in Von-Mises fitting'''
        return self.GetXY(Shifted=True,Radians=True,Normalized=True)

    def GetXY(self,Shifted=False,Radians=False,Normalized=False):
        ''' Get values of histogram.

        Shifted - If true, results come from -90 to 90, else 0 to 180
        Normalized - If true, angles are in radians and integral of histogram is 1
        '''
        Angles = self.X
        Values = self.Y

        if Shifted:
            Angles=np.delete(self.X,0)
            Values=np.delete(self.Y,0)
            RolingIndex=np.where(Angles==90)[0][0]+1
            Angles=np.roll(Angles,RolingIndex)
            Values=np.roll(Values,RolingIndex)
            Angles[Angles>90]-=180
            Angles = np.insert(Angles,0,-90)
            Values = np.insert(Values,0,Values[-1])
        
        if Radians:
            Angles=np.radians(Angles)

        if Normalized:
            IntensityIntegral = np.trapz(Values, x=Angles)
            Values = Values/IntensityIntegral

        return Angles,Values
        

    # Plot figure of intensity vs angle    
    def PlotHistogram(self,Show=False):
        FileName = self.OutputRoot if not Show else None
        XX,YY = self.X,self.Y
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.bar(XX,YY,width=1.0) #Histogram
        self.BasicPlotProperties(ax,"Angle","Intensity",[XX[0],XX[-1]])
        MATL.SaveMPLfig(fig,FileName)
        plt.close(fig)        
        
    # Plot figure of cumulative intensity vs angle 
    def PlotCumulative(self,Show=False):
        FileName = self.OutputRoot if not Show else None
        XX,YY = self.X,self.Y
        fig = plt.figure()
        ax=fig.add_subplot(111)
        XX=np.append(XX,180)
        YY=np.append([0],np.cumsum(YY))
        ax.plot(XX,YY)                #Cumulated Data
        ax.plot([XX[0],XX[-1]],[0,1]) #Straigt Line 0-1
        self.BasicPlotProperties(ax,"Angle","Cumulative Normalized Intensity",[XX[0],XX[-1]])
        MATL.SaveMPLfig(fig,FileName)
        plt.close(fig)   
    
    # Basic plotting properties
    def BasicPlotProperties(self,ax,xlabel,ylabel,xlim):
        xmin,xmax=xlim
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([0,1])
        ax.set_xticks(list(range(int(xmin),int(xmax+2),45)))
        ax.xaxis.grid(True,color='k',linestyle='--', linewidth=0.5)

class Fitting(object):
    def __init__(self,Angles,Intensities):
        self.Angles = Angles
        self.Intensities = Intensities
        self.parameters = None
        self.GOFTests = DF.GOFTests()#[NEW] 
        self.gofresults = None

    def getParameters(self):
        return self.parameters

    def getGOF(self):
        self.GOFTests.UpdateData(self.Angles,self.Intensities,self.parameters,self.N_VonMises,self.Uniform)#[NEW] 
        self.gofresults = self.GOFTests.computeGOF()#[New] 
        self.computeGOF()
        return self.gofresults

    # ======= #
    # Von-Mises Fitting

    def collectVMUParameters(self,theta):
        # Collect Parameters of Distributions
        p=[]     # Weight
        kappa=[] # Dispersion
        m=[]     # Direction
        for vmi in range(self.N_VonMises):
            p.append(theta[self.NPVM*vmi])
            kappa.append(theta[self.NPVM*vmi+1])
            m.append(theta[self.NPVM*vmi+2])
        pu = []

        if self.Uniform: pu = theta[-1]
        return p,kappa,m,pu

    def GeneralVMULogLike(self,theta):
        ''' General LogLike function
        theta[0 to 3*N_VonMises_ -1] - Parameters for Von-Mises
        theta[3*N_VonMises_] - Parameter for Uniform
        '''
        VMPDF_SCALING_PARAMETER = 0.5

        #Collect Args
        Angles_ = np.array(self.Angles).T
        Intensities_ = np.array(self.Intensities).T
        N_VonMises_ = self.N_VonMises
        Uniform_ = self.Uniform
        
        p_,kappa_,m_,pu_ = self.collectVMUParameters(theta)

        # Add Contributions of distributions
        fm_=np.zeros_like(Intensities_)
        for vmi in range(N_VonMises_):
            fm_ += p_[vmi] * sp.stats.vonmises.pdf( Angles_, 
                kappa_[vmi], loc = m_[vmi], scale = VMPDF_SCALING_PARAMETER )

        if Uniform_:
            Min_Angle = min(Angles_)
            Range_Angles = (max(Angles_) - min(Angles_))
            fm_ += pu_ * sp.stats.uniform.pdf( Angles_, loc=Min_Angle, scale=Range_Angles)

        # Return
        out = (-1.0)*(np.sum( np.multiply( np.log( fm_ ), Intensities_ ) ))
        return out
        
    def FitVMU(self,N_VonMises=1,Uniform=True):
        self.N_VonMises = N_VonMises
        self.Uniform = Uniform
        self.NPVM = 3 # Number of Von-Mises Parameters

        MIN_ANG = -np.pi/2
        MAX_ANG = np.pi/2

        N_Parameters = self.N_VonMises * self.NPVM
        uniform_p = 0.0
        if Uniform:
            N_Parameters += 1
            uniform_p = 0.1
        in_guess = np.zeros(N_Parameters)
        Mcons = np.zeros(N_Parameters)
        bnds = []

        for vmi in range(self.N_VonMises): #p,k,m
            in_guess[self.NPVM*vmi] = (1.0-uniform_p)/self.N_VonMises
            in_guess[self.NPVM*vmi+1] = np.array((6.0))
            in_guess[self.NPVM*vmi+2] = 0.5*(MIN_ANG + (MAX_ANG-MIN_ANG)*vmi/self.N_VonMises)
            Mcons[self.NPVM*vmi] = 1.0
            bnds.extend([(0., 1.), (0.1, 100.), (MIN_ANG, MAX_ANG)])
        if Uniform:
            in_guess[-1]=uniform_p
            Mcons[-1] = 1.0
            bnds.extend([(0.,0.8)])

        cons = ({'type': 'eq', 'fun': lambda x:  np.dot(x,Mcons)-1.0})

        # Run Optimization
        N_it=0
        while N_it<5: #Repeat when result converged to exactly the edge of the angle
            results = sp.optimize.minimize(self.GeneralVMULogLike, in_guess, \
                            method='SLSQP', bounds=bnds, constraints=cons, \
                            tol=1e-6, options={'maxiter': 100, 'disp': False})
            # Test if result is on edge 
            sol = results.x
            angindex = [self.NPVM*vmi+2 for vmi in range(self.N_VonMises)]
            MinLimIndx = [i for i in angindex if abs(sol[i]-MIN_ANG)<1E-3]
            MaxLimIndx = [i for i in angindex if abs(sol[i]-MAX_ANG)<1E-3]

            if MinLimIndx or MaxLimIndx:
                if MinLimIndx: sol[MinLimIndx]= MAX_ANG
                if MaxLimIndx: sol[MaxLimIndx]= MIN_ANG
                in_guess = sol
                N_it+=1
            else:
                N_it=5

        self.parameters = self.collectVMUParameters(results.x)  

        return self.parameters        

    def PlotVMU(self,ax=None):
        Y=np.zeros_like(self.Angles)
        p_,kappa_,m_,pu_ = self.parameters
        for vmi in range(self.N_VonMises): #p,k,m
            Y += p_[vmi] * sp.stats.vonmises.pdf( self.Angles, 
            kappa_[vmi], loc = m_[vmi], scale = 0.5 )
        if self.Uniform: Y+= pu_*sp.stats.vonmises.pdf( self.Angles, 
            1.0E-3, loc = 0.0, scale = 0.5 )

        axN=False
        if ax is None:
            fig = plt.figure()
            ax=fig.add_subplot(111)
            axN=True
        ax.bar(np.degrees(self.Angles),self.Intensities,width=1.0) 
        ax.plot(np.degrees(self.Angles),Y,color="r")
        ax.set_xlim([-90,90])
        ax.set_ylim([0,1.1*max(np.amax(self.Intensities),np.amax(Y))])
        ax.set_xlabel("Angles")
        ax.set_ylabel("Intensities")
        
        if axN: return fig

## ---------------------------------------------------------------------- #
## TEMPORARY COMMENT FOR TESTING; DELETE AFTER SUCCESSFUL TESTING
#    # ========================== #
#    # GOF tests (Dimitris Code)
#    # functions defined: 
##    """
##    computeGOF
##    Watson_GOF
##    Watson_CDF
##    Kuiper_GOF
##    Kuiper_CDF
##    CDFX
##    Get_PDF_VMU
##    Get_CDF_VMU
##    makePoints
##    Uniformity_test
##    ECDF
##    myR2
##    R2_GOF
##    Plot_PP_GOF
##    dispersion_coeff
##    """
#    
#    # ---------------------------------------------------------------------- #
#    def computeGOF(self):
#        """
#        Compute and Collect results from the following GOF tests: 
#            1. Watson
#            2. Kuiper
#            3. R2
#        Returns:
#        --------
#            self.gofresults
#        """
#        # Watson GOF test: 
#        watson_data = self.Watson_GOF( alphal=2 )
#        
#        # Kuiper GOF test: 
#        kuiper_data = self.Kuiper_GOF( )
#        
#        # R2 coefficient of determination: 
#        r2_data = self.R2_GOF( )
#        
#        # if you want to keep only the common columns from the data frames
#        #   of the tests, use the following names_ list: 
#        names_ = ['GOFTest', 'Symbol', 'Statistic', 'CriticalValue','PValue', \
#                  'SignifLevel', 'Decision']
#        
#        self.gofresults = pd.concat([watson_data, kuiper_data, r2_data])
#        self.gofresults = self.gofresults[names_]
#        
#    # ---------------------------------------------------------------------- #
#    def Watson_GOF( self, alphal=2 ):
#        """
#        From Book: "Applied Statistics: Using SPSS, STATISTICA, MATLAB and R"
#                    by J. P. Marques de Sa, Second Edition
#                    SPringer, ISBN: 978-3-540-71971-7 
#                    Chapter 10-Directional Statistics, Section 10.4.3 
#        Watson U2 test of circular data X.
#        Parameters:
#        -----------
#            self.CDFX(): the EXPECTED distribution (cdf) from the fitting.
#        Evaluates:
#        ----------
#            U2: the Watson's test statistic
#            US: the modified test statistic for known (loc, kappa)
#            UC: the critical value at ALPHAL (n=100 if n>100)
#            where:
#            ------
#            ALPHAL: 1->0.1; 2->0.05 (default); 3->0.025; 4->0.01; 5->0.005
#        Returns:
#        --------
#            Watson_gof_results: dataFrame collecting the above.
#        References: 
#        ----------
#            1. Equations (6.3.34), (6.3.36) of:
#                Mardia K. V., Jupp P. E., (2000)
#                "Directional Statistics", Wiley.
#                ISBN: 0-471-95333-4 
#            2. Equation (7.2.8) of:
#                Jammalamadaka S. Rao, SenGupta A. (2001), 
#                "Topics in Circular Statistics", World Scientific.
#                ISBN: 981-02-3778-2 
#        """
#        
#        V = self.CDFX()
#        n = len(V)
#        Vb = np.mean(V)
#        cc = np.arange(1., 2*n, 2)
#        
#        # The Watson's statistic: 
#        # Eq. (6.3.34) of Ref. [1]: 
#        U2 = np.dot(V, V) - np.dot(cc, V)/n + n*(1./3. - (Vb - 0.5)**2.)
#        
#        # The modified Watson's statistic (when both loc and kappa are known): 
#        # Eq. (6.3.36) of Ref. [1]: 
#        Us = (U2 - 0.1/n + 0.1/n**2)*(1.0 + 0.8/n)
#        
#        alpha_lev = np.array([0.1, 0.05, 0.025, 0.01, 0.005])
#        #              alpha=   =0.1     =0.05    =0.025   =0.01    =0.005
#        c = np.array([ [ 2,     0.143,	0.155,	0.161,	0.164,	0.165 ],
#                       [ 3,     0.145,	0.173,	0.194,	0.213,	0.224 ],
#                       [ 4,     0.146,	0.176,	0.202,	0.233,	0.252 ],
#                       [ 5,     0.148,	0.177,	0.205,	0.238,	0.262 ],
#                       [ 6,     0.149,	0.179,	0.208,	0.243,	0.269 ],
#                       [ 7,     	0.149,	0.180,  0.210,	0.247,	0.274 ],
#                       [ 8, 	    0.150,	0.181,	0.211,	0.250,	0.278 ],
#                       [ 9,     0.150,	0.182,	0.212,	0.252,	0.281 ],
#                       [ 10,	    0.150,	0.182,	0.213,	0.254,	0.283 ],
#                       [ 12,	    0.150,	0.183,	0.215,	0.256,	0.287 ],
#                       [ 14,    	0.151,	0.184,	0.216,	0.258,	0.290 ],
#                       [ 16,    	0.151,	0.184,	0.216,	0.259,	0.291 ],
#                       [ 18,    	0.151,	0.184,	0.217,	0.259,	0.292 ],
#                       [ 20,	    0.151,	0.185,	0.217,	0.261,	0.293 ],
#                       [ 30,    	0.152,	0.185,	0.219,	0.263,	0.296 ],
#                       [ 40,    	0.152,	0.186,	0.219,	0.264,	0.298 ],
#                       [ 50,	    0.152,	0.186,	0.220,	0.265,	0.299 ],
#                       [ 100,	0.152,	0.186,	0.221,	0.266,	0.301 ] ])
#
#        # compute the critical value, by linear interpolation: 
#        if n >= 100:
#            uc = c[-1, alphal]
#        else:
#            for i in np.arange(0, len(c)): 
#                if c[i, 0] > n:
#                    break
#            n1 = c[i-1, 0]
#            n2 = c[i, 0]
#            c1 = c[i-1, alphal]
#            c2 = c[i, alphal]
#            uc = c1 + (n - n1)*(c2 - c1)/(n2 - n1)
#        
#        # compute the p-values: 
#        pval2 = self.Watson_CDF( U2 )
#        pvals = self.Watson_CDF( Us )
#        
#        if U2 < uc:
#            H0_W = "Do not reject"
#        else:
#            H0_W = "Reject"
#        
#        Watson_gof_results = pd.DataFrame({'GOFTest': "Waston", \
#                                           'Symbol': "U2", \
#                                           'Statistic': U2, \
#                                           'StatisticStar': Us, \
#                                           'CriticalValue': uc, \
#                                           'PValue': pval2, \
#                                           'ModifiedPValue': pvals, \
#                                           'SignifLevel': alpha_lev[alphal], \
#                                           'Decision': [H0_W]})
#        
#        Watson_gof_results = Watson_gof_results[['GOFTest', 'Statistic', \
#                                                 'Symbol', \
#                                                 'StatisticStar', \
#                                                 'CriticalValue','PValue', \
#                                                 'ModifiedPValue', 'SignifLevel', \
#                                                 'Decision']]
#    
#        return Watson_gof_results
#        # return U2, Us, uc, pval2, pvals, H0_W, alpha_lev[alphal]
#    
#    # ---------------------------------------------------------------------- #
#    def Watson_CDF( self, x ):
#        """
#        Function to return the value of the CDF of the asymptotic Watson 
#        distribution at x: 
#        Parameters:
#        -----------
#            x: (scalar) value to compute the above CDF.
#        Returns:
#        --------
#            y: (scalar) the CDF of asymptotic Watson computed at x.
#        References: 
#        ----------
#            1. Equation (6.3.37) of:
#                Mardia K. V., Jupp P. E., (2000)
#                "Directional Statistics", Wiley.
#                ISBN: 0-471-95333-4 
#            2. Equation (7.2.10) of:
#                Jammalamadaka S. Rao, SenGupta A. (2001), 
#                "Topics in Circular Statistics", World Scientific.
#                ISBN: 981-02-3778-2 
#        """
#        epsi = 1e-10
#                
#        k = 1
#        y = 0
#        asum = 1
#        while asum > epsi:
#            md = ((-1)**(k-1))*np.exp(-2.*((np.pi)**2)*(k**2)*x)
#            y += 2.*md
#            k += 1
#            asum = abs(md)
#            
#        #print('Watson k iterations =',k)
#        return y
#    
#    # ---------------------------------------------------------------------- #
#    def Kuiper_GOF( self ):
#        """
#        cY_:  the probability distribution of the postulated model on ordered data
#        Kuiper Vn test of circular data X.
#        Parameters:
#        -----------
#            self.CDFX(): the EXPECTED distribution (cdf) from the fitting.
#        Evaluates:
#        ----------
#            Vn: the Kuiper's test statistic
#            pVn: the p-value for the statistic
#            Vns: the modified Kuiper's test statistic
#            pVns: the p-value for the modified statistic
#            Vc: the critical value at ALPHAL (n=100 if n>100)
#            alp_lev: the alpha level of significance
#            where:
#            ------
#            ALPHAL: 1->0.1; 2->0.05 (default); 3->0.025; 4->0.01 
#        Returns:
#        --------
#            Kuiper_gof_results: dataFrame collecting the above.
#        References: 
#        ----------
#            1. Equation (6.3.25) of:
#                Mardia K. V., Jupp P. E., (2000)
#                "Directional Statistics", Wiley.
#                ISBN: 0-471-95333-4 
#            2. Equation (7.2.2) of:
#                Jammalamadaka S. Rao, SenGupta A. (2001), 
#                "Topics in Circular Statistics", World Scientific.
#                ISBN: 981-02-3778-2 
#        """
#        cY_ = self.CDFX()
#        n = len(cY_)
#        sqn = np.sqrt(n)
#        
#        # prepare the Kuiper test: 
#        ii = np.arange(0, 1, 1/n)
#        vec1 = cY_ - ii
#        jj = np.arange(1/n, 1+1/n, 1/n)
#        vec2 = jj - cY_
#        
#        Dp = max(vec1)
#        Dm = max(vec2)
#        
#        # compute the Kuiper statistic Vn: 
#        Vn = sqn*(Dp + Dm)
#        # compute the Kuiper p-value: 
#        pVn = self.Kuiper_CDF( Vn, n )
#        
#        # modified Kuiper statistic Vns: 
#        # eq. (6.3.30) without the sqrt(n): 
#        # for n>=8, both Vn and Vns are very close. 
#        Vns = Vn * (1 + 0.155/sqn + 0.24/n)
#        pVns = self.Kuiper_CDF( Vns, n )
#        
#        # Significance levels: 
#        alpha = np.array([ 0.10, 0.05, 0.025, 0.01 ])
#        # The Vc provided below are the upper quantiles of Vns,
#        # Table 6.3 of Ref. [1].
#        Vc = np.array([ 1.620, 1.747, 1.862, 2.001 ])
#        
#        if Vns < min(Vc):
#            H0_K = "Do not reject"
#            dif_v = abs(Vn - Vc)
#            ind_v = np.argmin(dif_v)
#            #print('ind_v:',ind_v)
#            alp_lev = alpha[ind_v]
#            Vcc = Vc[ind_v]
#        else:
#            H0_K = "Reject"
#            Vcc = Vc[-1]
#            alp_lev = 1
#        
#        Kuiper_gof_results = pd.DataFrame({'GOFTest': "Kuiper", \
#                                           'Statistic': Vn, \
#                                           'Symbol': "Vn", \
#                                           'ModStatistic': Vns, \
#                                           'CriticalValue': Vcc, \
#                                           'PValue': pVn, \
#                                           'ModifiedPValue': pVns, \
#                                           'SignifLevel': alp_lev, \
#                                           'Decision': [H0_K]})
#        
#        Kuiper_gof_results = Kuiper_gof_results[['GOFTest', 'Statistic', \
#                                                 'Symbol', 'ModStatistic', \
#                                                 'CriticalValue','PValue', \
#                                                 'ModifiedPValue', \
#                                                 'SignifLevel', \
#                                                 'Decision']]
#        
#        #return Vn, Vcc, pVn, H0_K, alp_lev
#        return Kuiper_gof_results
#    
#    # ---------------------------------------------------------------------- #
#    def Kuiper_CDF( self, x, n ):
#        """
#        Function to return the value of the CDF of the asymptotic Kuiper
#        distribution at x: 
#        Parameters:
#        -----------
#            x: (scalar) value to compute the above CDF.
#            n: (integer) the number of points available from the observed data
#        Returns:
#        --------
#            y: (scalar) the CDF of asymptotic Kuiper computed at x.
#        References: 
#        ----------
#            1. Equation (6.3.29) of:
#                Mardia K. V., Jupp P. E., (2000)
#                "Directional Statistics", Wiley.
#                ISBN: 0-471-95333-4 
#            2. Equation (7.2.6) of:
#                Jammalamadaka S. Rao, SenGupta A. (2001), 
#                "Topics in Circular Statistics", World Scientific.
#                ISBN: 981-02-3778-2 
#        """
#        epsi = 1e-10
#        
#        x2 = x * x
#        g = -8.*x/(3.*np.sqrt(n))
#        k = 1
#        y = 0
#        asum = 1
#        while asum > epsi:
#            k2 = k * k
#            md1 = (4.*k2*x2 - 1.)*np.exp(-2.*k2*x2)
#            md2 = k2*(4.*k2*x2 - 3.)*np.exp(-2.*k2*x2)
#            y += 2.*md1 + g*md2
#            k += 1
#            asum = abs(md1+md2)
#            
#        #print('Kuiper k iterations =', k)
#        return y
#    
#    # ---------------------------------------------------------------------- #
#    def CDFX( self ):
#        """
#        Function to get the CDF of FITTED or EXPECTED distribution 
#            over the observed x-axis points.
#        Called in methods:
#            'Watson_GOF' 
#            'Kuiper_GOF' 
#        Parameters:
#        -----------
#            self.makePoints(): returns the observed points over which the CDFX
#                                will be evaluated.
#        Returns:
#        --------
#            cdfX: the requested CDF
#        """
#        p_X = self.makePoints()
#        aa = np.sort( p_X )
#        
#        cX_t = self.Get_CDF_VMU( aa )
#                
#        if max(cX_t) > 1:
#            cdfX = cX_t - (max(cX_t) - 1.)
#        else:
#            cdfX = cX_t
#        
#        return cdfX
#    
#    # ---------------------------------------------------------------------- #
#    def Get_PDF_VMU( self, x ):
#        """
#        Parameters:
#        -----------
#            x: any vector over which to compute the PDF of the mixture VMU 
#        
#        Returns:
#        --------
#            pdf_vmu: the PDF for the underlying mixture of Von Mises and 
#                     Uniform distributions, calculated over the vector x.
#        """
#        pdf_vmu = np.zeros_like(self.Angles)
#        p_, kappa_, m_, pu_ = self.parameters
#        #print(p_, kappa_, m_, pu_)
#        for vmi in range( self.N_VonMises ): # p, k, m 
#            pdf_vmu += p_[vmi] * sp.stats.vonmises.pdf( x, kappa_[vmi], \
#                                                 loc = m_[vmi], scale = 0.5 )
#        
#        if self.Uniform:
#            pdf_vmu += pu_ * sp.stats.vonmises.pdf( x, 1.0E-3, \
#                                                    loc = 0.0, scale = 0.5 )
#        
#        return pdf_vmu
#    
#    # ---------------------------------------------------------------------- #
#    def Get_CDF_VMU( self, x ):
#        """
#        Parameters:
#        -----------
#            x: any vector over which to compute the CDF of the mixture VMU 
#        
#        Returns:
#        --------
#            cdf_vmu: the CDF for the underlying mixture of Von Mises and 
#                     Uniform distributions, calculated over the vector x.
#        """
#        cdf_vmu = np.zeros_like(x)
#        p_, kappa_, m_, pu_ = self.parameters
#        for vmi in range( self.N_VonMises ): # p, k, m 
#            cdf_vmu += p_[vmi] * sp.stats.vonmises.cdf( x, kappa_[vmi], \
#                                                 loc = m_[vmi], scale = 0.5 )
#           
#        if self.Uniform:
#            cdf_vmu += pu_ * sp.stats.vonmises.cdf( x, 1.0E-3, \
#                                                    loc = 0.0, scale = 0.5 )
#        
#        return cdf_vmu
#    
#    # ---------------------------------------------------------------------- #
#    def makePoints( self ):
#        """
#        Function to convert the normalized intensities into points per angle
#        Parameters:
#        -----------
#        self.Angles :       bins of angles in degrees (usually 1 degree width)
#        self.Intensities :  normalized light intensity per bin angle 
#    
#        Returns:
#        --------
#        p_X : array-like, shape=(n_pop, )
#                col-1 : angles in rads of self.Angles 
#        """
#        Int = self.Intensities
#        Angs = np.degrees(self.Angles)
#        
#        # the equally-spaced segments in the space [-pi/2, pi/2]:
#        bin_angle = len( Int )
#        
#        # population factor:
#        sfact = 3.0
#        
#        # convert the intensity to points per angle:
#        #p_Xi = np.round( sfact*n_X[:,1] / max(n_X[:,1]) )
#        #p_Xi = np.round( sfact*(n_X[:,1]/min(n_X[:,1])) )
#        p_Xi = np.ceil( sfact*(Int/min(Int)) )
#        
#        # convert them to integer type:
#        p_Xi = p_Xi.astype(int)
#        
#        # the total population for all angles:
#        #p_Xi_sum = sum(p_Xi)
#        
#        # make the points: 
#        p_Xd = np.zeros((0,))
#        for i in range(bin_angle):
#            Xtemp = np.repeat(Angs[i], p_Xi[i])
#            p_Xd = np.append(p_Xd, Xtemp)
#        
#        # convert degrees to radians:
#        p_X = np.radians(p_Xd)
#            
#        return p_X
#    
#    # ---------------------------------------------------------------------- #
#    def Uniformity_test( self ):
#        '''
#        Assesement of Uniformity of Raw data:
#            using a uniform probability plot:
#                plot the sorted observations theta_i/pi against i(n+1)
#                If the data come from a uniform distribution, then the points
#                should lie near a straight line of unit slope passing through 
#                the origin.
#            The function also returns the R2 "coefficient of determination" 
#                for that fit: the closer to unity the more uniform the data.
#        '''
#        # get the observed data: 
#        pX = ( self.makePoints() + np.pi/2 )/np.pi
#        n = len(pX)
#        h = np.arange(1/(n+1), n/(n+1), 1/(n+1))
#        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#        ax.plot(h, pX[1::], color='r', linestyle=':', label='data')
#        #ax.plot([0, 1], [0, 1], color='g', linestyle='--', label='45$^o$ line')
#        ax.plot(h, h, color='g', linestyle='--', label='45$^o$ line')
#        ax.square()
#        ax.legend()
#        
#        # compute the R2 coefficient of determination: 
#        SSE = sum((h - pX[1::])**2)
#        SSTO = sum((h - np.mean(h))**2)
#        R2 = 1 - SSE/SSTO
#        ax.annotate('$R^2=%s$' % R2, xy=(0.4,1.0), xytext=(0.4, 1.0) )
#        
#        return R2
#
#    # ---------------------------------------------------------------------- #
#    def ECDF( self, data ):
#        """
#        CAUTION!: NOT for Light Intensity (from FFT) data! 
#        Function to evaluate the Empirical Distribution Function (ECDF) 
#        associated with the empirical measure of a sample. 
#        Parameters:
#        -----------
#            data: random sample, (rads) in the nature of the problem to consider 
#        Returns:
#        --------
#            x_values:   the x-axis of the ECDF figure (rads)
#            y_values:   the y-axis of the ECDF figure, the ECDF
#        called in method: 
#            'R2_GOF' 
#        """
#        # put the data into an array:
#        raw_data = np.array(data)
#        # create a sorted series of unique data:
#        cdfx = np.sort(np.unique(raw_data))
#        # raw_data = data
#        # # create a sorted series of unique data
#        # cdfx = np.sort(data)
#        # x-data for the ECDF: evenly spaced sequence of the uniques
#        x_values = np.linspace(start=min(cdfx),stop=max(cdfx),num=len(cdfx))
#        # size of the x_values
#        size_data = raw_data.size
#        # y-data for the ECDF:
#        y_values = []
#        for i in x_values:
#            # all the values in raw data less than the ith value in x_values
#            temp = raw_data[raw_data <= i]
#            # fraction of that value with respect to the size of the x_values
#            value = temp.size / size_data
#            # pushing the value in the y_values
#            y_values.append(value)
#            
#        # return both x and y values
#        return x_values, y_values
#    
#    # ---------------------------------------------------------------------- #
#    def myR2( self, Fexp, Fobs ):
#        """
#        Function to compute the coefficient of determination R2, 
#        a measure of goodness-of-fit.
#        Parameters:
#        -----------
#            Fexp: expected (postulated) CDF
#            Fobs: empirical (observed) CDF,
#                    can be computed from the function: 
#                        ECDF_Intensity( angles, values )
#        Returns:
#        --------
#            R2 
#        """
#        
#        Fbar = np.mean(Fexp)
#        
#        FobsmFexp = sum((Fobs - Fexp)**2)
#        
#        num = sum((Fexp - Fbar)**2)
#        
#        den = num + FobsmFexp
#        
#        R2 = num / den
#                
#        return R2
#    
#    # ---------------------------------------------------------------------- #
#    def R2_GOF( self ):
#        """
#        Function to prepare the data for the calculation of the R2 coefficient.
#        Returns:
#        --------
#            R2_results: dataFrame collecting R2 results.
#        """
#        # get the observed data: 
#        pX = self.makePoints()
#        #print('size pX: ', np.size(pX))
#        #print('pX1: ', pX[0:6])
#        # compute the empirical cumulative distribution function: 
#        x_obs, cdf_obs = self.ECDF( pX )        
#        
#        # compute the expexted probability distribution function for the fit:
#        fX_r2 = self.Get_PDF_VMU( x_obs )
#        
#        # compute the expexted cumulative distribution function for the fit:
#        dx = np.diff(x_obs)
#        cdfX_r2 = np.ones(len(x_obs),)
#        cdfX_r2[0:-1] = np.cumsum(fX_r2[0:-1]*dx)
#        
#        # compute the R2 coefficient: 
#        R2 = self.myR2( cdfX_r2, cdf_obs )
#        
#        # this is subjective: what should a good R2 be???
#        # print('R2 =', R2)
#        R2c = 0.90
#        if R2 > R2c:
#            H0_R2 = "Do not reject"
#        else:
#            H0_R2 = "Reject"
#        
#        R2_results = pd.DataFrame({'GOFTest': "R2", 'Statistic': R2, \
#                                   'Symbol': "R2", 'CriticalValue': R2c, \
#                                   'Decision': [H0_R2]})
#        
#        R2_results = R2_results[['GOFTest', 'Statistic', 'Symbol', \
#                                 'CriticalValue', 'Decision']]
#        
#        # plot the PP-plot: 
#        PPfig = self.Plot_PP_GOF( cdfX_r2, cdf_obs )
#        
#        return R2_results
#        #return R2, H0_R2
#    
#    # ---------------------------------------------------------------------- #
#    def Plot_PP_GOF( self, Fexp, Fobs ):
#        """
#        Generate the Probability-Probability plot.
#        Parameters:
#        -----------
#            Fexp: distribution function of postulated (expected) distribution
#            Fobs: empirical distribution function (the original data)
#        Ref: 
#        """
#        x = np.linspace(0,1,len(Fobs))
#        
#        x = np.linspace(0,1,len(Fobs))
#        
#        fig, ax = plt.subplots(1,1,figsize=(4,3))
#        ax.plot(Fexp, Fobs, 'k.', lw=2, alpha=0.6, label='P-P plot')
#        ax.plot(x, x, 'r--', lw=2, alpha=0.6, label='1:1')
#        ax.set_title('P-P plot for a Mixture Model')
#        ax.set_xlabel('Mixture distribution function')
#        ax.set_ylabel('Empirical distribution function')
#        ax.grid(color='gray', alpha=0.3, linestyle=':', linewidth=1)
#        ax.legend()
#        
#        return fig
#        
#    # ---------------------------------------------------------------------- #
#    def dispersion_coeff( self ):
#        """
#        Return the dispersion coefficient of one family of dispersed fibers,
#        as defined by Holpzapfel.
#        kappa_ip = ( 1 - I1(b) / I0(b) )/2      (eq. 2.23)
#        Ref: Holzapfel et al. (2015), "Modelling non-symmetric collagen fibre 
#            dispersion in arterial walls", J. R. Soc. Interface 12: 20150188 
#        """
#        # Collect the dispersion coefficients for every fiber family: 
#        dispersion = []
#        p_, kappa_, m_, pu_ = self.parameters
#        for vmi in range( self.N_VonMises ): # p, k, m 
#            kap_ = kappa_[vmi]
#            dispersion.append( 0.5*(1 - (sp.special.i1(kap_))/(sp.special.i0(kap_)) ) )
#        
#        return dispersion
#
#    # ========================== #
#    # End of GOF tests (Dimitris Code)
#    # ---------------------------------------------------------------------- #
#


# Main ...
if __name__ == "__main__":
    print("Testing Orientation Analysis")
    # BaseAngFolder="C:\\Users\\Miguel\\Work\\FiberOrientationHistogram\\AngleFiles"
    # TestImage="Tests\\OrientationAnalysis\\RWM_IE_Z.png"   
    
    # OAnalysis = OrientationAnalysis(BaseAngFolder=BaseAngFolder)
    # OAnalysis.SetImage(TestImage)
    # Results = OAnalysis.ApplyFFT()
    # Results.PlotHistogram()
    # OAnalysis.ApplyGradient().PlotHistogram()



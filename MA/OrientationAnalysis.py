#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from builtins import str
from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import os

import MA.ImageProcessing as MAIP
import MA.Tools as MATL

from joblib import Parallel, delayed
import multiprocessing



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
        
    def GetXY(self):
        return self.X,self.Y
        
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



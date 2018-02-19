#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from PIL import Image
import numpy as np
import scipy.optimize as spoptimize

import math
import progressbar
import os
from matplotlib import cm
import matplotlib.colors as cols
import MA.Tools as MATL



# === Conversion between formats of Matrices
def Image2np(Mpix):
    '''Convert image/matrix index style from Image to numpy.'''
    return np.transpose(np.flipud(Mpix))

def np2Image(Mpix):
    '''Convert image/matrix index style from numpy to Image.'''
    return np.flipud(np.transpose(Mpix))
    
# === Show Images    
def ShowImageRGB(Matrix,**kwargs):
    '''Show pixel data in matrix as RGB on the screen.'''
    if not type(Matrix[0,0]) == np.ndarray:
        Matrix=ConvertToRGB(Matrix,**kwargs)    
    Image.fromarray(Matrix,mode="RGBA").show()
       
def ShowImage(Matrix,resc=False):
    '''Show pixel data in matrix as 8-bit on the screen.'''
    if resc:
        Matrix=Rescale8bit(Matrix)
    Image.fromarray(np.uint8(Matrix),mode="L").show()

# === Save Images
def SaveImageRGB(Matrix,Name,**kwargs):
    '''Save pixel data in matrix as RGB on a file.'''
    Name=MATL.FixName(Name,'.png')
    if not type(Matrix[0,0]) == np.ndarray:
        Matrix=ConvertToRGB(Matrix,**kwargs)  
    Image.fromarray(Matrix,mode="RGBA").save(Name,"png")    
  
def SaveImage(Matrix,Name,resc=False):
    '''Save pixel data in matrix as 8-bit on a file.'''
    Name=MATL.FixName(Name,'.png')
    if resc:
        Matrix=Rescale8bit(Matrix)
    Image.fromarray(np.uint8(Matrix),mode="L").save(Name,"png")    

def SaveImageRaw(Matrix,Name):
    '''Save pixel data in matrix as 32-bit float on a file.'''
    Name=MATL.FixName(Name,'.tiff')
    Image.fromarray(np.float32(Matrix),mode="F").save(Name,"TIFF")   


# === Save/Show Images    
def SaveShowImage(Matrix,Root=None,Suffix=None):
    '''Save Image if Root is valid, otherwise show it on the screen.'''
    Name = MATL.MakeRoot(Root=Root,Suffix=Suffix)
    if Name is not None:
        SaveImage(Matrix,Name)
    else:
        ShowImage(Matrix)
    
# === Load Images
    
def GetRGBAImageMatrix(Name,Silent=False,**kwargs):
    ''' Get RGBA pixels from image as numpy array.''' 
    Img = Image.open(Name)
    if not Silent: print("Original Image mode: " + Img.mode)
    if not Img.mode == 'RGBA':
        Img=Img.convert("RGBA")
        if not Silent: print("Converted to 'RGBA' Mode")
    Mpix = np.copy(np.asarray(Img))
    Img.close()
    return Mpix    
    
def GetImageMatrix(Name,Silent=False,GetTiffRes=False, **kwargs):
    ''' Get 8-bit (or raw) pixels from image as numpy array.'''
    Ext = Name[-3:].lower()
    if Ext == 'raw':
        Img = OpenPILRaw(Name,**kwargs)
    else:
        Img = Image.open(Name)
        if not Silent: print("Original Image mode: " + Img.mode)
        if Img.mode in ['RGB','RGBA']:
            Img=Img.convert("L")
            if not Silent: print("Converted to 'L' Mode")

    if GetTiffRes and (Ext == 'tif' or Ext == 'iff'): xres,yres = Img.info['resolution']
    
    # sizes=Img.size
    # if not sizes[0]==sizes[1]: raise ValueError("Image must be a square")
    Mpix = np.copy(np.asarray(Img))
    Img.close()
    
    if GetTiffRes:
       if (Ext == 'tif' or Ext == 'iff'):
           return Mpix,(xres,yres)
       else:
           return Mpix,None
    return Mpix


def OpenPILRaw(FName,dims=None,PrecBits=None,ConvertL=False):
    '''Code to open a RAW image with PIL.'''
    with open(FName, "rb") as file:
        data = file.read()
        
    #Default 32-bit grayscale float, big endian
    wordsize=4 #size in bytes
    mode='F'
    moderaw='F;32BF'
    if PrecBits: #http://pillow.readthedocs.io/en/3.4.x/handbook/writing-your-own-file-decoder.html
        if PrecBits==8:
            wordsize=1 #size in bytes
            mode='L'
            moderaw='L'
        elif not PrecBits==32:
            print("WARNING: Precision not implemented. Used 32-bit float")

    ds=int(len(data)/wordsize) #Size of data (corrected for bytes/pixel)
    
    if not dims:
        d=int(0.5+math.sqrt(ds))
        if ds==d**2:
            dims=(d,d)
        else:
            raise ValueError("Image is not square! Set proper dims")
    else:
        if not dims[0]*dims[1]==ds:
            raise ValueError("Something wrong with dimensions given")
    
    myimg=Image.frombytes(mode, dims, data, "raw",moderaw, 0, 1)
    if ConvertL: myimg=myimg.convert("L")
    return myimg
    
    
def SmoothImage(Matrix,N=1):
    '''Smooths a Matrix with kernel [0 1 0; 1 4 1; 0 1 0]/8'''
    A=Matrix.copy()
    for i in range(N):
        s=A.shape[0]-1
        B=A*4.0
        B[-s:,:]+=A[:s,:]
        B[0,:]+=A[0,:]
        B[:,-s:]+=A[:,:s]
        B[:,0]+=A[:,0]

        B[:s,:]+=A[-s:,:]
        B[-1,:]+=A[-1,:]
        B[:,:s]+=A[:,-s:]
        B[:,-1]+=A[:,-1]
        B*=1/8
        A=B
    return A

# Convert 1D Image to RGBA
def ConvertToRGB(MpixO,cmap=None,markzero=False,nf=0.001,N=1024,RGBPoints=None):
    '''Convert 1D Image to RGBA'''

    Mpix=np.copy(MpixO)
    minp=np.amin(Mpix)
    maxp=np.amax(Mpix)
    if markzero: #Overides RGBPoints and cmap
        p=-minp/(maxp-minp)
        if p<0:
            raise ValueError("Data does not contain zeros, leading to inconsistent cmap")
        RGBPoints= [[0.0       ,1,0,0], [p-nf*p    ,1,0.8,0.8], [p-nf*p,1,1,0],
                    [p+(1-p)*nf,1,1,0], [p+(1-p)*nf,0.8,0.8,1], [1.0   ,0,0,1]]

    if RGBPoints is not None: #Overides cmap
        cmap = Getcmap(RGBPoints=RGBPoints,N=N)
        
    if cmap is None: cmap='jet'
        
    if type(cmap)==type(' '):
        cmap = Getcmap(Name=cmap,N=N)
        
    Mpix-=minp
    Mpix/=(maxp-minp)
    return cmap(Mpix, bytes=True)

def Getcmap(Name=None,RGBPoints=None,N=100):
    '''Create a color map.'''
    if Name is not None:
        return cm.get_cmap(Name,N)
    elif RGBPoints is not None:
        Pts=np.array(RGBPoints)
        Pts=Pts.reshape(-1,4)
        Ncol=Pts.shape[0]
        rtpl = []
        gtpl = []
        btpl = []
        for i in range(Ncol):
            rtpl.append( (Pts[i,0], Pts[i,1], Pts[i,1]) )
            gtpl.append( (Pts[i,0], Pts[i,2], Pts[i,2]) )
            btpl.append( (Pts[i,0], Pts[i,3], Pts[i,3]) )
        cdict = {'red':rtpl, 'green':gtpl, 'blue':btpl}
        return cm.get_cmap(cols.LinearSegmentedColormap(Name,cdict,N=N),N)
    else:
        raise("Error: Cannot Set-up Color Map")   


def ResizeMat(Mpix,S):
    '''Resize Image in matrix form.'''
    if S == np.shape(Mpix)[0]:
        return Mpix
    return np.maximum(0,np.asarray(Image.fromarray(np.float32(Mpix),mode="F").resize((S,S),Image.LANCZOS)))
        


def Rescale8bit(Matrix):
    '''Convert data in matrix to 8-bit by rescalling.'''
    min=np.amin(Matrix)
    max=np.amax(Matrix)
    return np.uint8((Matrix-min)/(max-min)*255+0.5)  

        

# TRANSFORMATIONS

class CoordsObj(object):
    def __init__(self,Coords=None,Img=None,XYScaling=None,ZScaling=None,InactiveThreshold=None):
        self.Mat = None
        self.XYScaling = None
        self.MN = None
        self.Coords = None
        
        if Coords is not None:
            self.setFromCoords(Coords)
        elif Img is not None:
            self.setFromImg(Img,XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold)

    # INITIALIZATION

    def setFromImg(self,Img,XYScaling=None,ZScaling=None,InactiveThreshold=None):
        ''' Create Coordinates object from Path or Numpy'''
        if type(Img) == np.ndarray:
            self.setFromMat(Img,XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold)
        elif type(Img) == type("string"):
            self.setFromPath(Img,XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold)

    def setFromPath(self,Img,XYScaling=None,ZScaling=None,InactiveThreshold=None):
        ''' Create Coordinates object from Image Path'''
        Mat , ImgXYScaling = GetImageMatrix(Img,GetTiffRes=True)
        if XYScaling is None: XYScaling=ImgXYScaling
        self.setFromMat(Mat,XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold)

    def setFromMat(self,Mat,XYScaling=None,ZScaling=None,InactiveThreshold=None):
        ''' Create Coordinates object from Matrix (image)'''
        self.Mat = Mat
        if ZScaling is not None: self.Mat*=ZScaling
        self.XYScaling=XYScaling
        if self.XYScaling is None: self.XYScaling = (1.0,1.0)
        self.MN=Mat.shape
        self.Coords=self.MatToCoords(InactiveThreshold=InactiveThreshold)

    def resetFromCoords(self,Coords):
        ''' Same as setFromCoords but also resets MN and XYScaling'''
        self.Mat = None
        self.XYScaling = None
        self.MN = None
        self.Coords = Coords

    def setFromCoords(self,Coords):
        ''' Create Coordinates object from Coordinates. Note that MN and XYScaling remain the same as before'''
        self.Mat = None
        self.Coords = Coords

    def fixXYScaling(self,XYScaling=None):
        if XYScaling is not None: self.XYScaling = XYScaling # Overide with new XYScale
        if self.XYScaling is None: self.XYScaling = (1.0,1.0) # Use Default XYScale

    def fixMN(self,MN=None):
        if MN is not None: self.MN = MN # Overide with new MN
        if self.MN is None: self.setFittedMN() # Use Default MN

    def setFittedMN(self):
        M=(np.amax(self.Coords[:,0])-np.amin(self.Coords[:,0]))/self.XYScaling[0]
        N=(np.amax(self.Coords[:,1])-np.amin(self.Coords[:,1]))/self.XYScaling[1]
        self.MN = (M,N)

    def fitDim(self,CoordObjInstance):
        self.XYScaling = CoordObjInstance.XYScaling
        self.MN = CoordObjInstance.MN

    def computeMat(self,MN=None,XYScaling=None):
        ''' Computes matrix from coordinates '''
        self.fixXYScaling(XYScaling)
        self.fixMN(MN)
        self.Mat = self.CoordsToMat()

    # GET DATA

    def getCoords(self):
        ''' Gets Coordinates '''
        return self.Coords

    def getTransformedCoords(self,R,T):
        return self.ApplyRotationAndTranslation(self.Coords,R,T)

    def getMat(self,MN=None,XYScaling=None):
        ''' Gets Matrix'''
        if (self.Mat is None) or (MN != self.MN) or (XYScaling != self.XYScaling):
            self.computeMat(MN=MN,XYScaling=XYScaling)
        return self.Mat

    def getTransformedMat(self,R,T,MN=None,XYScaling=None):
        if (self.Mat is None) or (MN != self.MN) or (XYScaling != self.XYScaling):
            self.fixXYScaling(XYScaling)
            self.fixMN(MN)
            return self.CoordsToMat(R=R,T=T)

        
        
    def getIJ(self,R=None,T=None):
        if R is None: R=np.array([0,0,0])
        if T is None: T=np.array([0,0,0])

        NewCoords = self.ApplyRotationAndTranslation(self.Coords,R,T)

        M,N=self.MN
        sx,sy = self.XYScaling
        IJ=np.zeros((NewCoords.shape[0],2))
        X=IJ[:,0]
        Y=IJ[:,1]
        X[:]=NewCoords[:,0].copy()
        Y[:]=NewCoords[:,1].copy()
        Z=NewCoords[:,2].copy()

        #Scale Image
        X/=sx
        Y/=sy

        # Center the image
        X+=M/2
        Y+=N/2

        #Fix out of bounds
        IX=np.logical_and(X>=0,X<=M-1)
        IY=np.logical_and(Y>=0,Y<=N-1)
        II=np.logical_and(IX,IY)
    
        return IJ[II],Z[II]

    # STATIC METHODS FOR COORD TRANSFORMATION

    @staticmethod
    def getSingleRotationMatrix(ang,posone):
        '''
        Makes a 3x3 matrix where the posone position has a 1 in the diagonal
        and the remaining two positions have a [[c,s],[-s,c]] matrix
        '''
        c, s = np.cos(ang), np.sin(ang)
        M=np.eye(3)
        indx=[0,1,2]
        indx.remove(posone)
        M[np.ix_(indx,indx)]=np.array([[c,-s],[s,c]])
        return M
    
    @classmethod
    def getRotationMatrix(cls,rV):
        ''' Get Rotation matrix after providing the 3 angles (phi,theta,psi) '''
        phi,theta,psi = rV
        A=cls.getSingleRotationMatrix(phi,0)
        B=cls.getSingleRotationMatrix(-theta,1)
        C=cls.getSingleRotationMatrix(psi,2)
        return np.matmul(C,np.matmul(B,A))
    
    @classmethod
    def ApplyRotationAndTranslation(cls,Coords,rV,tV):
        ''' Returns the Coords after being transformed '''
        RM = cls.getRotationMatrix(rV)
        return np.dot(Coords,np.transpose(RM))+tV


    # CONVERSION

    def MatToCoords(self,InactiveThreshold=None):
        '''Convert image matrix to active coordinates

        Optional: InactiveThreshold (float) - Values less or equal to this threshold are considered inactive
        Output: Array of [x,y,z] 
        '''
        sx,sy=self.XYScaling
        # Get Shape of Array (image)
        M,N=self.MN
        # Use coordinates wrt center
        dx,dy=M/2,N/2
        # Compute values of x and y
        B1, B0 = np.meshgrid(range(N), range(M))
        B0=(B0-dx)*sx
        B1=(B1-dy)*sy
        # Make coordinates
        Coords=np.dstack((B0, B1, self.Mat))
        # Return active coordinates
        if InactiveThreshold is None:
            return Coords[self.Mat==self.Mat]
        else:
            return Coords[self.Mat>InactiveThreshold]

    def CoordsToMat(self,R=None,T=None):
        '''Convert points in "Coordinate" form to matrix form.
        '''
        Mat=np.zeros(self.MN)
        IJ,Z=self.getIJ(R=R,T=T)
        IJ=np.round(IJ).astype(np.int32)

        Mat[IJ[:,0],IJ[:,1]]=Z[:]

        return Mat


class ImageFit(object):
    def __init__(self, RefImage, **kwargs):
        '''Class to find the transformation that fits
           an image to a reference image.

        Input: RefImage - Image (String-Path, ndarray-Matrix)
        Optional: XYScaling - dimensions/pixel of image (sx,sy)
                   ZScaling - Factor to scale the pixel values (z)
        '''

        self.CReference = CoordsObj(Img=RefImage, **kwargs)

        self.CImage = CoordsObj()
        self.CImage.fitDim(self.CReference)


    def FitNewImage(self,Image,silent=False, **kwargs):
        ''' Load an image and fit it to the Reference Image

        Input: Image to fit - Image (String-Path, ndarray-Matrix)
        Optional: XYScaling (scalar,scalar) - Tuple with the scaling (dimension/pixel) of the X and Y directions
                  ZScaling (scalar) - Scalar with the scaling for the pixel value corresponding to the Z direction
                  InactiveThreshold (scalar) - Scalar with the Z value below which the pixel will be ignored
        '''
        self.CImage = CoordsObj(Img=Image, **kwargs)
        self.CImage.fitDim(self.CReference)
        return self.Solve(silent=silent)

    def FitNewCoords(self,Coords,silent=False):
        ''' Fit Coords to the Reference Image

        Input: Image to fit - Image (String-Path, ndarray-Matrix)
        Optional: XYScaling (scalar,scalar) - Tuple with the scaling (dimension/pixel) of the X and Y directions
                  ZScaling (scalar) - Scalar with the scaling for the pixel value corresponding to the Z direction
                  InactiveThreshold (scalar) - Scalar with the Z value below which the pixel will be ignored
        '''
        self.CImage.setFromCoords(Coords)
        return self.Solve(silent=silent)
    
    # SOLVING

    def Solve(self,silent=False):
        # Get initialization parameters
        self.RT0a = self.getRTarr(self.GetInitialization())
        # R,T = self.GradientDescent(self.RT0a)
        R,T = self.ScipyOptimize(self.RT0a,silent=silent)
        return R,T


    def GetInitialization(self):
        ''' Get Initialization of comparison of two membranes.

        Input: Coordinate format of active points
        Output: Initial rotation angles and translation vector
        '''

        CRef = self.CReference.getCoords()
        CNew = self.CImage.getCoords()

        # Estimate Center of Membranes
        O_Ref=np.average(CRef,axis=0)
        O_New=np.average(CNew,axis=0)
    
        # Estimate Highest point of Membranes
        M_Ref=CRef[np.argmax(CRef[:,2])]
        M_New=CNew[np.argmax(CNew[:,2])]

        # Estimate Relative rotation between two membranes
        r0=[0,0,-self.ComputeHorizontalAngle(M_New-O_New,M_Ref-O_Ref)]
        # Estimate Relative displacement between two membranes
        t0=O_Ref-CoordsObj.ApplyRotationAndTranslation(O_New,r0,[0,0,0])
    
        return r0,t0

    # METHODS

    @staticmethod
    def ComputeHorizontalAngle(A,B):
        '''Compute angle in xy plane between two vectors
        '''
        ax,ay,_=A/np.linalg.norm(A)
        bx,by,_=B/np.linalg.norm(B)
        return -np.arctan2(ax*by-bx*ay,ax*bx+ay*by)

    @staticmethod
    def getRTtup(X):
        return X[0:3],X[3:6]
    
    @staticmethod
    def getRTarr(tup):
        return np.concatenate(tup)

    def CostFunction(self,RT):

        R,T=self.getRTtup(RT)
        IJNew,ZNew = self.CImage.getIJ(R,T)
    
        IJi=np.floor(IJNew).astype(np.int32)
        IJf=np.ceil(IJNew).astype(np.int32)
        alf=(IJNew-IJi)/(IJf-IJi)
        Zii=self.CReference.Mat[IJi[:,0],IJi[:,1]]
        Zif=self.CReference.Mat[IJi[:,0],IJf[:,1]]
        Zfi=self.CReference.Mat[IJf[:,0],IJi[:,1]]
        Zff=self.CReference.Mat[IJf[:,0],IJf[:,1]]

        Zai=Zfi*alf[:,0]-Zii*(alf[:,0]-1)
        Zaf=Zff*alf[:,0]-Zif*(alf[:,0]-1)
        Zab=Zaf*alf[:,1]-Zai*(alf[:,1]-1)

        DZ=ZNew-Zab
        DZ=DZ**2
        return np.sum(DZ)

    def ComputeGradientCost(self,RT):
        R,T=self.getRTtup(RT)
        LR,LT = len(R),len(T)

        Cost0 = self.CostFunction(RT)
    
        DR=1E-8
        DT=1E-8
        GradientR = np.zeros(LR)
        GradientT = np.zeros(LT)

        #Rotations Gradient
        for i in range(LR):
            deltaR = np.zeros(LR)
            deltaR[i] = DR
            NRT=self.getRTarr((R+deltaR,T))
            CostNew = self.CostFunction(NRT)
            GradientR[i] = (CostNew-Cost0)/DR

        #Translations Gradient
        for i in range(LT):
            deltaT = np.zeros(LT)
            deltaT[i] = DT
            NRT=self.getRTarr((R,T+deltaT))
            CostNew = self.CostFunction(NRT)
            GradientT[i] = (CostNew-Cost0)/DT

        return self.getRTarr((GradientR,GradientT))

    def GradientDescent(self,RTin):
        GammaR=1E-10
        GammaT=1E-10
        Rin,Tin = self.getRTtup(RTin)

        R,T=np.array(Rin),np.array(Tin)
        R_old,T_old=R.copy(),T.copy()
        LR,LT = len(R),len(T)
        GR,GT = np.zeros(LR),np.zeros(LT)
        GR_old,GT_old = np.zeros(LR),np.zeros(LT)

        print("\n\n\n\n\n")
        Cost_new = self.CostFunction(self.getRTarr((R,T)))
     
        for i in range(150):
            GR_old,GT_old=GR,GT
            GR,GT = self.getRTtup(self.ComputeGradientCost(self.getRTarr((R,T))))

            if i>0:
                dr,dgr=R-R_old,GR-GR_old
                denR=np.dot(dgr,dgr)
                GammaR = max(np.dot(dr,dgr)/denR,1E-17)
                GammaR = np.dot(dr,dgr)/denR

                dt,dgt=T-T_old,GT-GT_old
                denT=np.dot(dgt,dgt)
                GammaT = max(np.dot(dt,dgt)/denT,1E-17)
                GammaT = np.dot(dt,dgt)/denT

                Gamma=((np.dot(dr,dgr)+np.dot(dt,dgt))/(denR+denT))
                GammaR=Gamma
                GammaT=Gamma
                #print(GammaR,GammaT)


            R_old,T_old=R,T

            R=R-GammaR*GR
            T=T-GammaT*GT
            #print(R,T)
            Cost_old=Cost_new
            Cost_new=self.CostFunction(self.getRTarr((R,T)))
            # print(Cost_old-Cost_new)

        return R,T

    def ScipyOptimize(self,RTin,silent=False):
        options=None
        if not silent: options={"disp":True}
        OptResult = spoptimize.minimize(self.CostFunction,RTin,jac=self.ComputeGradientCost,options=options)
        # OptResult = spoptimize.minimize(self.CostFunction,RTin,options={"disp":True})
        if not silent: print(OptResult.message)
        R,T=self.getRTtup(OptResult.x)
        return R,T










        
# SPECIAL IMAGES

# Make image with fibers given by raised cosine for ang and freq    
def CosAng(i,j,ang=0,freq=20):
    M,N  = np.shape(i)
    Angr=ang*np.pi/180
    x=i/M-1/2
    y=j/N-1/2
    eta=-np.sin(Angr)*x + np.cos(Angr)*y
    return (np.cos(eta*2*np.pi*freq)+1)/2

def MakeCosImage(MN,**kwargs):
    return np.fliplr(np.transpose(np.fromfunction(CosAng,MN,**kwargs)))

# Make image of a semi-sphere
def ShperePix(i,j,R=None):
    M,N  = np.shape(i)
    if R is None: R=min(M,N)/2
    x=i-M/2
    y=j-N/2
    rad=R**2-x**2-y**2
    rad[rad<0]=0
    return np.sqrt(rad)

def MakeSphereImage(MN,**kwargs):
    return np.fliplr(np.transpose(np.fromfunction(ShperePix,MN,**kwargs)))

def DistancePL(Point,Line):
    a,b = np.array(Line)
    p = np.moveaxis(np.array(Point), 0, -1)
    T = np.linalg.norm(b-a)
    n = (b-a)/T
    pa=p-a
    pb=p-b

    A = np.dot(pa,n)
    aa=n[np.newaxis,np.newaxis,:]*A[:,:,np.newaxis]

    d=pa-aa
    d[A<0]=pa[A<0]
    d[A>T]=pb[A>T]

    return np.linalg.norm(d,axis=2)


def RPix(i,j,d=0.1):
    M,N  = np.shape(i)
    if d is None: d=0.1
    x=2*i/(M-1)-1 # x in [-1,1]
    y=2*j/(N-1)-1 # y in [-1,1]
    L1=[(-3*d,-7*d),(-3*d,7*d)]
    L2=[(-3*d,7*d),(3*d,4*d)]
    L3=[(3*d,4*d),(-3*d,1*d)]
    L4=[(-3*d,1*d),(3*d,-7*d)]
    
    D=np.ones_like(x)
    D+=1.0
    for Line in [L1,L2,L3,L4]:
        D=np.minimum(D,DistancePL((x,y),Line))
    
    D[D>d]=d
    V=-D**2
    V+=d**2

    return np.sqrt(V)

def MakeRImage(MN,**kwargs):
    return np2Image(np.fromfunction(RPix,MN,**kwargs))
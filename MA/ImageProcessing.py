#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.optimize as spoptimize
from scipy.signal import convolve2d
from pyquaternion import Quaternion

import math
import progressbar
import os
from matplotlib import cm
import matplotlib.colors as cols
import MA.Tools as MATL

# === Operate on images
def FlipImage(NameIn,NameOut):
    Ext = NameIn[-3:].lower()
    isTif = (Ext == 'tif' or Ext == 'iff')
    Img = Image.open(NameIn)
    FlippedImage = Img.transpose(Image.FLIP_LEFT_RIGHT)

    if isTif:
        xres,yres = Img.info['resolution']
        kwargsdict={282:xres, 283:yres, 270: 'ImageJ=1.51w\nunit=micron\n',296:1}
        FlippedImage.save(NameOut,tiffinfo=kwargsdict)
    else: 
        FlippedImage.save(NameOut)



### Note on image matrix formats ###
# Consider an image given by:
#   00 10 20
#   15 25 35
#   30 40 50 
# There are two formates for storing this image in a matrix.
#  1) Image format - Typical image formats usually store a matrix such that positions (0,0), (1,0), (1,2) would be 00, 15, 35.
#  2) Coord format - This format will behave as if indexes are coordinates, such that positions (0,0), (1,0), (1,2) would be 30, 40, 10.

# === Conversion between formats of Matrices
def Image2np(Mpix):
    '''Convert matrix format from Image to Coord.'''
    return np.transpose(np.flipud(Mpix))

def np2Image(Mpix):
    '''Convert matrix format from Coord to Image.'''
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

def SaveImageRaw(Matrix,Name,TifRes=None,resolution=None):
    '''Save pixel data in matrix as 32-bit float on a file.'''
    Name=MATL.FixName(Name,'.tif')
    Image.fromarray(np.float32(Matrix),mode="F").save(Name,"TIFF",dpi=resolution)   

def SaveTif(Matrix,Name,resolution=None):
    '''Save pixel data in matrix as 32-bit float on a tif file.'''
    Name=MATL.FixName(Name,'.tif')
    MyImage = Image.fromarray(np.float32(Matrix),mode="F")
    if resolution:
        xres,yres = resolution
        kwargsdict={282:xres, 283:yres, 270: 'ImageJ=1.51w\nunit=micron\n',296:1}
        MyImage.save(Name,tiffinfo=kwargsdict)
    else: 
        MyImage.save(Name)

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

    if GetTiffRes and (Ext == 'tif' or Ext == 'iff'):
        try:
            xres,yres = Img.info['resolution']
        except:
            try:
                xres,yres = Img.info['dpi']
            except:
                print("No Resolution Info")
                xres=1.0
                yres=1.0


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
    
# TRANSFORM IMAGES
    
def ScaleTif(TifSource,TifTarget,ScaleFactor):
    ''' TifTarget = TifSource * ScaleFactor'''
    Mpix,res = GetImageMatrix(TifSource,Silent=True,GetTiffRes=True)
    Mpix*=ScaleFactor
    SaveTif(Mpix,TifTarget,resolution=res)
    
def SmoothImage(Matrix,N=1):
    '''Smooths a Matrix with kernel [0 1 0; 1 4 1; 0 1 0]/8'''
    A=Matrix.copy()
    for _ in range(N):
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

def MaskedSmooth(Matrix,N=1):
    '''
    Matrix - Image to smooth
    N - number of iterations
    '''
    kernel = np.add.outer(*2*(np.arange(3)%2,))**2 / 8
    aux_kernel = np.arange(9).reshape(3, 3)%2 / 8

    Mask = Matrix==0

    corrector = convolve2d(Mask, aux_kernel, 'same')
    result = Matrix.copy()
    for j in range(N):
        result = result * corrector + convolve2d(result, kernel, 'same')
        result[Mask]=0
    return result



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
    def __init__(self,Coords=None,Img=None,XYScaling=None,ZScaling=None,InactiveThreshold=None,Smooth=None):
        self.Mat = None # Matrix of image in Coord format
        self.XYScaling = None
        self.MN = None
        self.Coords = None
        
        if Coords is not None:
            self.setFromCoords(Coords)
        elif Img is not None:
            self.setFromImg(Img,XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold,Smooth=Smooth)

    # INITIALIZATION

    def setFromImg(self,Img,XYScaling=None,ZScaling=None,InactiveThreshold=None,Smooth=None):
        ''' Create Coordinates object from Path or Numpy Matrix in Image format'''
        if type(Img) == np.ndarray:
            self.setFromMat(Image2np(Img),XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold,Smooth=Smooth)
        elif type(Img) == type("string"):
            self.setFromPath(Img,XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold,Smooth=Smooth)

    def setFromPath(self,Img,XYScaling=None,ZScaling=None,InactiveThreshold=None,Smooth=None):
        ''' Create Coordinates object from Image Path'''
        Mat , ImgXYScaling = GetImageMatrix(Img,GetTiffRes=True)
        if XYScaling is None: XYScaling=ImgXYScaling
        self.setFromMat(Image2np(Mat),XYScaling=XYScaling,ZScaling=ZScaling,InactiveThreshold=InactiveThreshold,Smooth=Smooth)

    def setFromMat(self,Mat,XYScaling=None,ZScaling=None,InactiveThreshold=None,Smooth=None):
        ''' Create Coordinates object from Numpy Matrix in Coord format'''
        self.Mat = Mat
        if ZScaling is not None: self.Mat*=ZScaling
        self.XYScaling=XYScaling
        if self.XYScaling is None: self.XYScaling = (1.0,1.0)
        self.MN=Mat.shape
        self.Coords=self.MatToCoords(InactiveThreshold=InactiveThreshold,Smooth=Smooth)

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
        M=(np.amax(self.Coords[:,0])-np.amin(self.Coords[:,0]))*self.XYScaling[0]
        N=(np.amax(self.Coords[:,1])-np.amin(self.Coords[:,1]))*self.XYScaling[1]
        self.MN = (M,N)

    def fitDim(self,CoordObjInstance):
        self.XYScaling = CoordObjInstance.XYScaling
        self.MN = CoordObjInstance.MN

    def computeMat(self,MN=None,XYScaling=None):
        ''' Computes matrix from coordinates '''
        self.fixXYScaling(XYScaling)
        self.fixMN(MN)
        self.Mat = self.CoordsToMat()

    def saveAsImage(self,ImageName):
        SaveTif(np2Image(self.getMat().copy()),ImageName,self.XYScaling)


    # GET DATA

    def getCoords(self):
        ''' Gets Coordinates '''
        return self.Coords

    def getTransformedCoords(self,R,T,pavg=None):
        return self.ApplyRotationAndTranslation(self.Coords.copy(),R,T,pavg=pavg)

    def getMat(self,MN=None,XYScaling=None,GetResolution=False):
        ''' Gets Matrix'''
        if (self.Mat is None) or (MN != self.MN) or (XYScaling != self.XYScaling):
            self.computeMat(MN=MN,XYScaling=XYScaling)
        if GetResolution: return self.Mat,self.XYScaling
        return self.Mat

    def getTransformedMat(self,R,T,pavg=None,MN=None,XYScaling=None,GetResolution=False):
        if (self.Mat is None) or (MN != self.MN) or (XYScaling != self.XYScaling):
            self.fixXYScaling(XYScaling)
            self.fixMN(MN)
            Mat=self.CoordsToMat(R=R,T=T,pavg=pavg)
            if GetResolution: return Mat,self.XYScaling
            return Mat

    def getIJ(self,R=None,T=None,pavg=None):
        if R is None: R=np.array([0,0,0])
        if T is None: T=np.array([0,0,0])

        NewCoords = self.getTransformedCoords(R,T,pavg=pavg)

        M,N=self.MN
        #sx,sy = float(self.XYScaling[0]),float(self.XYScaling[1])
        sx,sy = 1/float(self.XYScaling[0]),1/float(self.XYScaling[1])
        IJ=np.zeros((NewCoords.shape[0],2))
        X=IJ[:,0]
        Y=IJ[:,1]
        X[:]=NewCoords[:,0].copy()
        Y[:]=NewCoords[:,1].copy()
        Z=NewCoords[:,2].copy()

        #Scale Image
        X/=sx
        Y/=sy

        # (OLD) Center the image
        # X+=M/2
        # Y+=N/2

        #Fix out of bounds
        IX=np.logical_and(X>=0,X<=M-1)
        IY=np.logical_and(Y>=0,Y<=N-1)
        II=np.logical_and(IX,IY)
    
        return IJ[II],Z[II]#,NewCoords[II]


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
        ''' Get Rotation matrix after providing the 3 angles (phi,theta,psi) or a quaternion or a new system of coordinates (A,B,C)'''
        if len(rV)==3:
            phi,theta,psi = rV
            A=cls.getSingleRotationMatrix(phi,0)
            B=cls.getSingleRotationMatrix(-theta,1)
            C=cls.getSingleRotationMatrix(psi,2)
            return np.matmul(C,np.matmul(B,A))
        elif len(rV)==4:
            return Quaternion(rV).rotation_matrix

    @classmethod
    def rotationMatrixToEulerAngles(cls,RM):
        sy = math.sqrt(RM[0,0] * RM[0,0] +  RM[1,0] * RM[1,0])
        singular = sy < 1e-6
    
        if  not singular :
            x = math.atan2(RM[2,1] , RM[2,2])
            y = math.atan2(-RM[2,0], sy)
            z = math.atan2(RM[1,0], RM[0,0])
        else :
            x = math.atan2(-RM[1,2], RM[1,1])
            y = math.atan2(-RM[2,0], sy)
            z = 0
    
        return np.array([x, y, z])

    @classmethod
    def ApplyRotationAndTranslation(cls,Coords,rV,tV,pavg=None):
        ''' Returns the Coords after being transformed. Rotations are relative to pavg position (default [0,0,0])'''

        # Return if no transformation
        if np.linalg.norm(rV)==0.0 and np.linalg.norm(tV)==0.0:
            return Coords

        # Robust input of pavg
        if type(pavg) is np.ndarray:
            pass
        elif pavg=="auto":
            pavg=np.average(Coords,axis=0)
        elif type(pavg) is list:
            pavg=np.array(pavg)
        elif (pavg is None) or (pavg == 0):
            if pavg is None: print("Roation Center Assumed at Origin. Set pavg=0 to supress this message")
            pavg=np.zeros(3)
        else:
            raise ValueError("Can't Understand the value given for pavg:"+str(pavg))

        # Get rotation Matrix
        RM = cls.getRotationMatrix(rV)

        # Get Relative Coordinates wrt Center of Gravity (pavg)
        Cg=Coords-pavg
        # Get Rotated Relative Coordinates
        RCg=np.dot(Cg,np.transpose(RM))

        return RCg+(pavg+tV)

    # CONVERSION
    def MatToCoords(self,InactiveThreshold=None,Smooth=None):
        '''Convert image matrix to active coordinates

        Optional: InactiveThreshold (float) - Values less or equal to this threshold are considered inactive
        Output: Array of [x,y,z] 
        '''
        sx,sy = 1/float(self.XYScaling[0]),1/float(self.XYScaling[1])
        # Compute values of x and y
        Nr,Nc= self.MN
        x = np.linspace(0, (Nc-1), Nc, dtype=np.float32)
        y = np.linspace(0, (Nr-1) , Nr, dtype=np.float32)
        B0, B1 = np.meshgrid(x, y, sparse=False, indexing='ij')
        B0*=sx
        B1*=sy
        # Make coordinates
        newmat=self.Mat.copy()
        if Smooth is not None:
            newmat=MaskedSmooth(newmat,N=Smooth)

        Coords=np.dstack((B0, B1, newmat))
        # Return active coordinates
        if InactiveThreshold is None:
            return Coords[self.Mat==self.Mat]
        else:
            return Coords[self.Mat>InactiveThreshold]

    def CoordsToMat(self,R=None,T=None,pavg=None):
        '''Convert points in "Coordinate" form to matrix form.
        '''
        Mat=np.zeros(self.MN)
        IJ,Z=self.getIJ(R=R,T=T,pavg=pavg)
        IJ=np.round(IJ).astype(np.int32)

        Mat[IJ[:,0],IJ[:,1]]=Z[:]

        return Mat


class ImageFit(object):
    def __init__(self, RefImage, InactiveThreshold=None, **kwargs):
        '''Class to find the transformation that fits
           an image to a reference image.

        Input: RefImage - Image (String-Path, ndarray-Matrix)
        Optional: XYScaling - dimensions/pixel of image (sx,sy)
                   ZScaling - Factor to scale the pixel values (z)
        '''

        self.CReference = CoordsObj(Img=RefImage, InactiveThreshold=InactiveThreshold, **kwargs)
        self.RefInactiveThreshold=InactiveThreshold
        self.CImage = CoordsObj()
        self.CImage.fitDim(self.CReference)
        self.AverageDZ = 0.0


    def FitNewImage(self,Image,silent=False, **kwargs):
        ''' Load an image and fit it to the Reference Image

        Input: Image to fit - Image (String-Path, ndarray-Matrix)
        Optional: XYScaling (scalar,scalar) - Tuple with the scaling (dimension/pixel) of the X and Y directions
                  ZScaling (scalar) - Scalar with the scaling for the pixel value corresponding to the Z direction
                  InactiveThreshold (scalar) - Scalar with the Z value below which the pixel will be ignored
        '''
        self.CImage = CoordsObj(Img=Image, InactiveThreshold=self.RefInactiveThreshold, **kwargs)
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
        # R,T = self.getRTtup(self.RT0a)
        R,T = self.ScipyOptimize(self.RT0a,silent=silent)

        RM = CoordsObj.getRotationMatrix(R)
        TZero = T + self.O_New - np.dot(self.O_New,np.transpose(RM))
        return R,TZero


    def GetInitialization(self):
        ''' Get Initialization of comparison of two membranes.

        Input: Coordinate format of active points
        Output: Initial rotation angles and translation vector
        '''

        CRef = self.CReference.getCoords()
        CNew = self.CImage.getCoords()

        # Estimate Center of Membranes
        O_Ref=np.average(CRef,axis=0)
        self.O_Ref = O_Ref
        O_New=np.average(CNew,axis=0)
        self.O_New = O_New

        # Estimate Highest point of Membranes
        M_Ref=CRef[np.argmax(CRef[:,2])]
        M_New=CNew[np.argmax(CNew[:,2])]

        # Estimate Relative rotation between two membranes
        rotangle=self.ComputeHorizontalAngle(M_New-O_New,M_Ref-O_Ref)

        # Angles
        r0 = [0,0,-rotangle]
        t0 = O_Ref - O_New

        # Quaternions
        # r0=Quaternion(axis=[0,0,1],radians=-rotangle).elements
        # t0=O_Ref-CoordsObj.ApplyRotationAndTranslation(O_New,r0,[0,0,0])
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
        if len(X)==6:
            return X[0:3],X[3:6]
        elif len(X)==7:
            return X[0:4],X[4:7]
        
    @staticmethod
    def getRTarr(tup):
        return np.concatenate(tup)

    def CostFunction(self,RT,printdetails=False):
        R,T=self.getRTtup(RT)
        IJNew,ZNew = self.CImage.getIJ(R,T,pavg=self.O_New)

        IJi=np.floor(IJNew).astype(np.int32)
        IJf=np.ceil(IJNew).astype(np.int32)

        posisInt = IJf==IJi
        IJf[posisInt]+=1

        alf=(IJNew-IJi)/(IJf-IJi)
        Zii=self.CReference.Mat[IJi[:,0],IJi[:,1]]
        Zif=self.CReference.Mat[IJi[:,0],IJf[:,1]]
        Zfi=self.CReference.Mat[IJf[:,0],IJi[:,1]]
        Zff=self.CReference.Mat[IJf[:,0],IJf[:,1]]
        
        Iii=Zii>self.RefInactiveThreshold
        Iif=Zif>self.RefInactiveThreshold
        Ifi=Zfi>self.RefInactiveThreshold
        Iff=Zff>self.RefInactiveThreshold
        II=np.logical_and(np.logical_and(Iii,Iif),np.logical_and(Ifi,Iff))

        Zai=Zfi*alf[:,0]-Zii*(alf[:,0]-1)
        Zaf=Zff*alf[:,0]-Zif*(alf[:,0]-1)
        Zab=Zaf*alf[:,1]-Zai*(alf[:,1]-1)

        DZ=ZNew[II]-Zab[II]
        N=len(DZ)
        self.AverageDZ = np.sum(np.abs(DZ))/N
        DZ=DZ**2
        # C1=np.sum(DZ)/N # Average SSQ distance between doubly-active points of membranes (best fit)
        C1=self.AverageDZ # Average distance between doubly-active points of membranes (best fit)
        
        PI=1-N/len(ZNew) #percentage of nodes not contributing to fit cost
        if printdetails: print("len(ZNew)=",str(len(ZNew)),", len(II)="+str(N),"P1="+str(PI))
        Iai=Ifi*alf[:,0]-Iii*(alf[:,0]-1)
        Iaf=Iff*alf[:,0]-Iif*(alf[:,0]-1)
        Iab=Iaf*alf[:,1]-Iai*(alf[:,1]-1)
        III=np.logical_or(np.logical_or(Iii,Iif),np.logical_or(Ifi,Iff))
        PI=1-np.sum(Iab[III])/len(ZNew)
        C2=max(0,np.exp(10*(PI-0.1))-1)

        if printdetails:
            print("{0:12.6f}".format(PI),"{0:12.6f}".format(C1),"{0:12.6f}".format(C2),np.round(R,4),np.round(T,4))
        return C1 #+C2

    def ComputeGradientCost(self,RT,Incs=None):
        R,T=self.getRTtup(RT)
        if Incs is None:
            DR,DT=1e-9,1e-9
        else:
            DR,DT=Incs[0],Incs[1]
        LR,LT = len(R),len(T)

        Cost0 = self.CostFunction(RT,printdetails=False)
    
        GradientR = np.zeros(LR)
        GradientT = np.zeros(LT)

        #Rotations Gradient
        for i in range(LR):
            deltaR = np.zeros(LR)
            deltaR[i] = DR
            NRT=self.getRTarr((R+deltaR,T))
            CostNew = self.CostFunction(NRT,printdetails=False)
            GradientR[i] = (CostNew-Cost0)/DR

        #Translations Gradient
        for i in range(LT):
            deltaT = np.zeros(LT)
            deltaT[i] = DT
            NRT=self.getRTarr((R,T+deltaT))
            CostNew = self.CostFunction(NRT,printdetails=False)
            GradientT[i] = (CostNew-Cost0)/DT

        return self.getRTarr((GradientR,GradientT))
    
    def ScipyOptimize(self,RTin,silent=False):
        options=None
        if not silent: options={"disp":True}
        # options={"maxiter":0}
        OptResult = spoptimize.minimize(self.CostFunction,RTin,jac=self.ComputeGradientCost,options=options)
        # OptResult = spoptimize.minimize(self.CostFunction,RTin,jac=self.ComputeGradientCost,method='BFGS',options=options)
        # OptResult = spoptimize.minimize(self.CostFunction,RTin,method='L-BFGS-B',options=options)

        # OptResult = spoptimize.basinhopping(self.CostFunction,RTin,minimizer_kwargs={"jac":self.ComputeGradientCost})
        if not silent: print(OptResult.message)
        R,T=self.getRTtup(OptResult.x)
        return R,T

    def GradientDescent(self,RTin):
        Gamma=1E-10
        Rin,Tin = self.getRTtup(RTin)
        R,T=np.array(Rin),np.array(Tin)
        LR,LT = len(R),len(T)
        GR,GT = np.zeros(LR),np.zeros(LT)

        Cost_new = self.CostFunction(self.getRTarr((R,T)))
        R_old,T_old = R, T
        for i in range(15):
            
            # Gradient
            GR_old,GT_old=GR,GT
            GR,GT = self.getRTtup(self.ComputeGradientCost(self.getRTarr((R,T))))
            
            # Gamma
            if i>0: Gamma = self.AdjustGamma(R,T,R_old,T_old,GR,GT,GR_old,GT_old)
            Gamma = self.LineSearchSimple(R_old,T_old,Gamma,GR,GT)

            # Update
            R_old,T_old=R,T
            R,T=R-Gamma*GR,T-Gamma*GT
            Cost_old=Cost_new
            Cost_new=self.CostFunction(self.getRTarr((R,T)))
            dc=1-Cost_new/Cost_old
            print("Percent Reduction:",dc)
            if dc<1E-5: break

        return R,T

    def AdjustGamma(self,R,T,R_old,T_old,GR,GT,GR_old,GT_old):
        dr,dgr=R-R_old,GR-GR_old
        denR=np.dot(dgr,dgr)
        dt,dgt=T-T_old,GT-GT_old
        denT=np.dot(dgt,dgt)
        return ((np.dot(dr,dgr)+np.dot(dt,dgt))/(denR+denT))

    def LineSearchSimple(self,Ro,To,Gamma,dr,dt):
        def cost(a): return self.CostFunction(self.getRTarr((Ro-a*Gamma*dr,To-a*Gamma*dt)))
        CO = cost(0.0)

        alpha=1E-2
        ct=0
        while ct<20:
            if ct==0:
                print("Line Search",end="\n")
            else:
                print(".",end="\n")
            ct+=1
            NC=cost(alpha)
            if NC<CO:
                alpha=2*alpha
                CO=NC
            else:
                alpha=alpha/2
                break
        print("")
        return alpha*Gamma



        




# ADDITIONAL DRAWING FUNCTIONS IN PIL

def drawCircle(Draw,xy,diameter,**kwargs):
    x, y =xy
    diameter = int(diameter)-1
    r = int(diameter/2)
    c = diameter%2

    ltrb = [ x-r , y-r , x+r+c , y+r+c]
    Draw.ellipse(ltrb, **kwargs)

def draw_ellipse(image, bounds, width=3, outline='white', antialias=2):
    """Improved ellipse drawing function, based on PIL.ImageDraw.
    
    TODO: 1- Use Draw as input instead of image. 2- For speed only do AA on the ellipse instead of the whole image.
    """

    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results. 
    mask = Image.new(
        size=[int(dim * antialias) for dim in image.size],
        mode='L', color='black')
    draw = ImageDraw.Draw(mask)

    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
        left, top = [(value + offset) * antialias for value in bounds[:2]]
        right, bottom = [(value - offset) * antialias for value in bounds[2:]]
        draw.ellipse([left, top, right, bottom], fill=fill)

    # downsample the mask using PIL.Image.LANCZOS 
    # (a high-quality downsampling filter).
    mask = mask.resize(image.size, Image.LANCZOS)
    # paste outline color to input image through the mask
    image.paste(outline, mask=mask)

def drawArrow(Draw,xyi,vec,scale=100,width=0.13,color='black',tip="rounded"):
    ''' Draw an arrow given an origin point and a vector direction'''
    xyi=np.array(xyi)
    vec=np.array(vec)
    L=np.linalg.norm(vec)
    s=vec[1]/L
    c=vec[0]/L
    RM=np.array([[c,-s],[s,c]])

    angle=45
    sa,ca = np.sin(np.radians(angle)),np.cos(np.radians(angle))
    dl=1/3

    P0_R = np.array([0,0])
    P1_R = np.array([1,0])
    P2_R = np.array([1-ca*dl,sa*dl])
    P3_R = np.array([1-ca*dl,-sa*dl])

    P0 = tuple(RM@P0_R*L*scale + xyi)
    P1 = tuple(RM@P1_R*L*scale + xyi)
    P2 = tuple(RM@P2_R*L*scale + xyi)
    P3 = tuple(RM@P3_R*L*scale + xyi)

    wd = int(scale*width)
    Draw.line([P0,P1],fill=color,width=wd)
    Draw.line([P2,P1],fill=color,width=wd)
    Draw.line([P3,P1],fill=color,width=wd)

    if tip=="rounded":
        drawCircle(Draw,P0,wd,fill=color)
        drawCircle(Draw,P1,wd,fill=color)
        drawCircle(Draw,P2,wd,fill=color)
        drawCircle(Draw,P3,wd,fill=color)
    else:
        P4_R = np.ceil(P1_R*L*scale + np.array([sa,ca])*((wd-1)/2)) - np.array([1,0])
        P5_R = np.ceil(P1_R*L*scale + np.array([1/sa,0])*((wd-1)/2)) - np.array([1,0])
        P6_R = np.ceil(P1_R*L*scale + np.array([sa,-ca])*((wd-1)/2)) - np.array([1,0])
        P1x = tuple(np.array(P1) - np.array([1,0]))
        P4 = tuple(RM@P4_R + xyi)
        P5 = tuple(RM@P5_R + xyi)
        P6 = tuple(RM@P6_R + xyi)
        Draw.polygon([P1x,P4,P5,P6], fill=color)


def drawTextWithOutline(Draw,xy,text,font,text_color='white',outl_width=3,outl_color='black'):
    x,y = xy
    
    Draw.text((x+0*outl_width, y+outl_width),text,outl_color,font=font)
    Draw.text((x+outl_width, y-0*outl_width),text,outl_color,font=font)
    Draw.text((x-outl_width, y+0*outl_width),text,outl_color,font=font)
    Draw.text((x-0*outl_width, y-outl_width),text,outl_color,font=font)
    
    s=np.sqrt(2)
    Draw.text((x+outl_width/s, y+outl_width/s),text,outl_color,font=font)
    Draw.text((x+outl_width/s, y-outl_width/s),text,outl_color,font=font)
    Draw.text((x-outl_width/s, y+outl_width/s),text,outl_color,font=font)
    Draw.text((x-outl_width/s, y-outl_width/s),text,outl_color,font=font)
    
    Draw.text((x, y),text,text_color,font=font)

def drawScalebar(Draw,barLength=200,scale=1.0,font=None):
    ''' Draw Scale bar.

        barLength - Length in world coordinates'''
    
    cornerdist = int(0.93*2048)
    scalebartext = str(barLength)+" microns"
    scalebar = int(barLength/scale)

    xtext,ytext = Draw.textsize(scalebartext,font=font)
    Draw.rectangle([cornerdist-scalebar,cornerdist,cornerdist,cornerdist-scalebar/10],fill=(20,20,20),outline='black')
    Draw.text((cornerdist-scalebar/2-xtext/2, cornerdist+ytext/10),scalebartext,'black',font=font)


def MarkPointsInImage(ImageName,OutputName,Data,radius=10,offset=8,fontSize=48):
    ColSequence = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'] #http://colorbrewer2.org
    BImg = Image.open(ImageName)
    Draw = ImageDraw.Draw(BImg)
    font = ImageFont.truetype("ariblk.ttf", fontSize)

    scale = 1400/2048.0 #microns/pixel

    for index, row in Data.iterrows():
        x_raw,y_raw= row["X_T"],row["Y_T"]
        label = row["Position"]
        avm = np.radians(row["AVM_flat"])
        x,y = x_raw/scale,2048-y_raw/scale
      
        #print("Draw Arrows")
        drawArrow(Draw,(x,y),(np.cos(avm),-np.sin(avm)))  
        #print("Make Ellipse")
        Draw.ellipse((x-radius+1, y-radius+1, x+radius, y+radius), fill=ColSequence[index])
        draw_ellipse(BImg,(x-radius+1, y-radius+1, x+radius, y+radius), outline='black')
        #print("Draw Text")
        drawTextWithOutline(Draw,(x+offset,y+offset),label,font,text_color=ColSequence[index])

    #print("Draw Scalebar")
    scalebarfont = ImageFont.truetype("ariblk.ttf", int(fontSize*0.8))
    drawScalebar(Draw,barLength=200,scale=scale,font=scalebarfont)

    BImg.save(OutputName)







        
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
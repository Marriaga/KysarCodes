#!/usr/bin/python
from __future__ import division
from PIL import Image
import numpy as np
import math
import progressbar
import os
from matplotlib import cm
import matplotlib.colors as cols
import MAPyLibs.Tools as MATL

# Convert between image type of coordinates and array type of coordinates
def Image2np(Mpix):
    return np.transpose(np.flipud(Mpix))

def np2Image(Mpix):
    return np.flipud(np.transpose(Mpix))

    
#Show pixel data as RGB on the screen  
def ShowImageRGB(Matrix,**kwargs):
    if not type(Matrix[0,0]) == np.ndarray:
        Matrix=ConvertToRGB(Matrix,**kwargs)    
    Image.fromarray(Matrix,mode="RGBA").show()
    
#Show pixel data in matrix on the screen    
def ShowImage(Matrix,resc=False):
    if resc:
        Matrix=Rescale8bit(Matrix)
    Image.fromarray(np.uint8(Matrix),mode="L").show()

#Save pixel data as RGB on a file    
def SaveImageRGB(Matrix,Name,**kwargs):
    Name=MATL.FixName(Name,'.png')
    if not type(Matrix[0,0]) == np.ndarray:
        Matrix=ConvertToRGB(Matrix,**kwargs)  
    Image.fromarray(Matrix,mode="RGBA").save(Name,"png")    

#Save pixel data in matrix on a file    
def SaveImage(Matrix,Name,resc=False):
    Name=MATL.FixName(Name,'.png')
    if resc:
        Matrix=Rescale8bit(Matrix)
    Image.fromarray(np.uint8(Matrix),mode="L").save(Name,"png")    
    
# Save Image if Root is valid, otherwise show it on the screen    
def SaveShowImage(Matrix,Root=None,Suffix=None):
    Name = MATL.MakeRoot(Root=Root,Suffix=Suffix)
    if Name is not None:
        SaveImage(Matrix,Name)
    else:
        ShowImage(Matrix)
    

# Smooth a matrix in pixel format    
def SmoothImage(Matrix,N=1):
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
    Mpix=np.copy(MpixO)
    minp=np.amin(Mpix)
    maxp=np.amax(Mpix)
    if markzero: #Overides RGBPoints and cmap
        p=-minp/(maxp-minp)
        if p<0:
            raise("Data does not contain zeros, leading to inconsistent cmap")
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

    
# Create a color map
def Getcmap(Name=None,RGBPoints=None,N=100):
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


# Resize Image in matrix form
def ResizeMat(Mpix,S):
    if S == np.shape(Mpix)[0]:
        return Mpix
    return np.maximum(0,np.asarray(Image.fromarray(np.float32(Mpix),mode="F").resize((S,S),Image.LANCZOS)))
        
# Code to open a RAW image
def OpenPILRaw(FName,dims=None,PrecBits=None):
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
            raise ValueErkror("Something wrong with dimensions given")
    
    return Image.frombytes(mode, dims, data, "raw",moderaw, 0, 1)    

#Obtain the RGBA pixel information of an image in a numpy array
def GetRGBAImageMatrix(Name,**kwargs):
    Img = Image.open(Name)
    print("Original Image mode: " + Img.mode)
    if not Img.mode == 'RGBA':
        Img=Img.convert("RGBA")
        print("Converted to 'RGBA' Mode")
    Mpix = np.copy(np.asarray(Img))
    Img.close()
    
    return Mpix    
    
#Obtain the pixel information of an image in a numpy array
def GetImageMatrix(Name,**kwargs):
    if Name[-3:] == 'raw':
        Img = OpenPILRaw(Name,**kwargs)
    else:
        Img = Image.open(Name)
        print("Original Image mode: " + Img.mode)
        if Img.mode in ['RGB','RGBA']:
            Img=Img.convert("L")
            print("Converted to 'L' Mode")

    sizes=Img.size
    #if not sizes[0]==sizes[1]: raise ValueError("Image must be a square")
    Mpix = np.copy(np.asarray(Img))
    Img.close()
    
    return Mpix

 

  
#Convert data in matrix to 8-bit by rescalling    
def Rescale8bit(Matrix):
    min=np.amin(Matrix)
    max=np.amax(Matrix)
    return np.uint8((Matrix-min)/(max-min)*255+0.5)  

#Do FFT, compute power spectrum and return rotated.    
def GetPowerSpectrum(Matrix,s=None,rot=True):
    FFT = np.fft.fftshift(np.fft.fft2(Matrix,s=s))
    POW = np.real(FFT*np.conj(FFT))
    if rot: POW = np.flipud(np.transpose(POW))
    return POW

#Set the center of the Power Spectrum to 0 inside a circle of radius R    
def MaskCenter(Matrix,R):
    S=np.shape(Matrix)[0]
    def MkCircle(i,j,S,R):
        x=i-S/2
        y=j-S/2
        R2=R**2
        return x**2+y**2<=R2
    Matrix[np.flipud(np.transpose(np.fromfunction(MkCircle,(S,S),S=S,R=R)))]=0

#Scalled Hamming
def Hamming(z,alph=0.5):
    z=np.minimum(1/2,np.maximum(-1/2,z))
    return alph-(1-alph)*np.cos(2*np.pi*z+np.pi)    

    
def Hamming2DBuilder(i,j,alph=0.5,Htype="Seperable",shp=None):
    M ,N  = np.shape(i)
    LM,LN = ((M,N) if shp is None else shp)
    zi=(i/(M-1)-1/2)*M/LM
    zj=(j/(N-1)-1/2)*N/LN
    
    if Htype=="Seperable":
        return Hamming(zi,alph)*Hamming(zj,alph)
    elif Htype == "Radial":
        zr=np.sqrt(zi**2+zj**2)
        return Hamming(zr,alph)
    else:
        raise ValueError("2D Hamming type must be 'Seperable' or 'Radial'")        

def ApplyHamming(Mpix,(M,N),**kwargs):
    Hamm2D=np.fromfunction(Hamming2DBuilder,(M,N),**kwargs)
    return Mpix*Hamm2D
        
        
        
# SPECIAL IMAGES


#Get gauss distribution value     
def Gauss(x,s=1,D=1,ori=0.0):
    if s==1 and D==1:
        amp=0.5641895835477562869480794515607725858
    else:
        amp=1.0*D/(s*math.sqrt(math.pi))
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

        
def Rings(i,j,R=5,**kwargs):
    M,N  = np.shape(i)
    x=i-M/2
    y=j-N/2
    r=np.sqrt(x**2+y**2)
    dr=r-R
    return Gauss(dr,**kwargs)



# Make image with fibers given by raised cosine for ang and freq    
def CosAng(i,j,ang=0,freq=20):
    M,N  = np.shape(i)
    Angr=ang*np.pi/180
    x=i/M-1/2
    y=j/N-1/2
    eta=-np.sin(Angr)*x + np.cos(Angr)*y
    return (np.cos(eta*2*np.pi*freq)+1)/2

def MakeCosImage((M,N),**kwargs):
    return np.fliplr(np.transpose(np.fromfunction(CosAng,(M,N),**kwargs)))


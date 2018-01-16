#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from PIL import Image
import numpy as np
import math
import progressbar
import os
from matplotlib import cm
import matplotlib.colors as cols
import MAPyLibs.Tools as MATL



# === Conversion between formats of Matrices
def Image2np(Mpix):
    '''Convert image/matrix index style from  to numpy.'''
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
    print("Original Image mode: " + Img.mode)
    if not Img.mode == 'RGBA':
        Img=Img.convert("RGBA")
        if not Silent: print("Converted to 'RGBA' Mode")
    Mpix = np.copy(np.asarray(Img))
    Img.close()
    return Mpix    
    
def GetImageMatrix(Name,Silent=False,**kwargs):
    ''' Get 8-bit (or raw) pixels from image as numpy array.'''
    if Name[-3:] == 'raw':
        Img = OpenPILRaw(Name,**kwargs)
    else:
        Img = Image.open(Name)
        if not Silent: print("Original Image mode: " + Img.mode)
        if Img.mode in ['RGB','RGBA']:
            Img=Img.convert("L")
            if not Silent: print("Converted to 'L' Mode")

    sizes=Img.size
    #if not sizes[0]==sizes[1]: raise ValueError("Image must be a square")
    Mpix = np.copy(np.asarray(Img))
    Img.close()
    
    return Mpix

def OpenPILRaw(FName,dims=None,PrecBits=None):
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
            raise ValueErkror("Something wrong with dimensions given")
    
    return Image.frombytes(mode, dims, data, "raw",moderaw, 0, 1)     
    
    
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

        
        
# SPECIAL IMAGE

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


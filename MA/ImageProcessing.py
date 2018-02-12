#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from PIL import Image
import numpy as np
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
            raise ValueErkror("Something wrong with dimensions given")
    
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

def getSingleRotationMatrix(ang,posone):
    '''
    Makes a 3x3 matrix where the posone position has a 1 in the diagonal
    and the remaining two positions have a [[c,s],[-s,c]] matrix
    '''
    c, s = np.cos(ang), np.sin(ang)
    M=np.eye(3)
    indx=[0,1,2]
    indx.remove(posone)
    M[np.ix_(indx,indx)]=np.array([[c,s],[-s,c]])
    return M

def getRotationMatrix(rV):
    phi,theta,psi = rV
    A=getSingleRotationMatrix(phi,2)
    B=getSingleRotationMatrix(theta,0)
    C=getSingleRotationMatrix(psi,2)
    return np.matmul(C,np.matmul(B,A))

def ApplyRotationAndTranslation(Coords,rV,tV):
    RM = getRotationMatrix(rV)
    return np.dot(Coords,np.transpose(RM))+tV

def MatToCoords_old(mat,sx=1.0,sy=1.0,sz=1.0,center=False,GetZero=False):
    M,N=mat.shape
    if center:
        dx,dy=M/2,N/2
    else:
        dx,dy=0,0

    B1, B0 = np.meshgrid(range(N), range(M))
    B0=(B0-dx)*sx
    B1=(B1-dy)*sy

    Coords=np.dstack((B0, B1, mat*sz))

    if GetZero:
        return Coords,mat<=0
    else:
        return Coords

def MatToCoords(mat,sx=1.0,sy=1.0,sz=1.0):
    # Get Shape of Array (image)
    M,N=mat.shape
    # Use coordinates wrt center
    dx,dy=M/2,N/2
    # Compute values of x and y
    B1, B0 = np.meshgrid(range(N), range(M))
    B0=(B0-dx)*sx
    B1=(B1-dy)*sy
    # Make coordinates
    Coords=np.dstack((B0, B1, mat*sz))
    # Return non-zero coordinates
    return Coords[mat>0]


def CoordsToMat_old(Coords,center=True,resc=True,ZI=None):
    '''Convert points in "Coordinate" to matrix form.

    Input: Coordinates matrix
    Optional Input:
       - center (True/False): Coordinates are relative to center of matrix instead of 0,0 point
       - resc (True/False): Make all points fit into the matrix by rescaling x,y
       - ZI (List of indices): Identifies all indices that are "Backgound" and will have a value of 0
    '''
    M,N,_=Coords.shape
    Mat=np.zeros((M,N))
    X=Coords[:,:,0].copy()
    Y=Coords[:,:,1].copy()
    Z=Coords[:,:,2].copy()
    
    #Fit all image in image frame
    if resc==True:
        xmin,xmax=np.amin(X),np.amax(X)
        ymin,ymax=np.amin(Y),np.amax(Y)
        X-=xmin
        Y-=ymin
        X*=((M-1)/(xmax-xmin))
        Y*=((N-1)/(ymax-ymin))

    if center:
        X+=M/2
        Y+=N/2

    #Convert indexes to integers
    X=np.around(X).astype(np.int32)
    Y=np.around(Y).astype(np.int32)

    #Fix out of bounds
    IX=np.logical_and(X>=0,X<M)
    IY=np.logical_and(Y>=0,Y<N)
    II=np.logical_and(IX,IY)

    #Fix Background Zs
    if ZI is not None:
        Z[ZI]=0

    #Make image
    Mat[X[II],Y[II]]=Z[II]

    return Mat

def CoordsToMat(Coords,MN):
    '''Convert points in "Coordinate" to matrix form.

    Input: Coordinates matrix
           MN - Shape of output matrix
    '''
    M,N=MN
    Mat=np.zeros((M,N))
    X=Coords[:,0].copy()
    Y=Coords[:,1].copy()
    Z=Coords[:,2].copy()

    # Center the image
    X+=M/2
    Y+=N/2

    #Convert indexes to integers
    X=np.around(X).astype(np.int32)
    Y=np.around(Y).astype(np.int32)

    #Fix out of bounds
    IX=np.logical_and(X>=0,X<M)
    IY=np.logical_and(Y>=0,Y<N)
    II=np.logical_and(IX,IY)

    #Make image
    Mat[X[II],Y[II]]=Z[II]

    return Mat

def ComputeHorizontalAngle(A,B):
    '''Compute angle in xy plane between two vectors
    '''
    norm=np.linalg.norm
    ax,ay,az=A/norm(A)
    bx,by,bz=B/norm(B)
    return -np.arctan2(ax*by-bx*ay,ax*bx+ay*by)

def GetInitialization(MRef,MNew):
    ''' Get Initialization of comparison of two membranes.

    Input: numpy format of reference image and new image
    Output: Initial rotation angles and translation vector
    '''

    #Get Coordinate Arrays
    CRef,ZIRef=MatToCoords(MRef,center=True,GetZero=True)
    CNew,ZINew=MatToCoords(MNew,center=True,GetZero=True)
    
    #Get Active Points
    active_Ref=CRef[np.logical_not(ZIRef)]
    active_New=CNew[np.logical_not(ZINew)]

    # Estimate Center of Membranes
    O_Ref=np.average(active_Ref,axis=0)
    O_New=np.average(active_New,axis=0)
    
    # Estimate Highest point of Membranes
    M_Ref=active_Ref[np.argmax(active_Ref[:,2])]
    M_New=active_New[np.argmax(active_New[:,2])]

    # Estimate Relative rotation between two membranes
    r0=[ComputeHorizontalAngle(M_New-O_New,M_Ref-O_Ref),0,0]
    # Estimate Relative displacement between two membranes
    t0=O_Ref-ApplyRotationAndTranslation(O_New,r0,[0,0,0])
    
    return r0,t0

def GetInitialization2(CRef,CNew):
    ''' Get Initialization of comparison of two membranes.

    Input: coordinate format of active points
    Optional Input: ZIRef,ZINew (List of indices): Identifies all indices that are "Backgound"
    Output: Initial rotation angles and translation vector
    '''
    
    print("A")
    # Estimate Center of Membranes
    O_Ref=np.average(CRef,axis=0)
    O_New=np.average(CNew,axis=0)
    print("A")
    
    # Estimate Highest point of Membranes
    M_Ref=CRef[np.argmax(CRef[:,2])]
    M_New=CNew[np.argmax(CNew[:,2])]
    print("A")

    # Estimate Relative rotation between two membranes
    r0=[ComputeHorizontalAngle(M_New-O_New,M_Ref-O_Ref),0,0]
    # Estimate Relative displacement between two membranes
    t0=O_Ref-ApplyRotationAndTranslation(O_New,r0,[0,0,0])
    
    return r0,t0







        
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
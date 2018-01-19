#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import inspect
import shutil
import timeit
import glob
import csv
import os


### File and Folder Manipulation ###

# Deletes Folder Tree
def DeleteFolderTree(Folder):
    if os.path.exists(Folder):
        #TODO: add exception handling for read-only files
        shutil.rmtree(Folder)

# Combine Root and Suffix for a path. It's robust
def MakeRoot(Root=None,Suffix=None):
    if Root is not None:
        Fold,File= os.path.split(Root)
        if Suffix is None and File=="":
            Name = Fold
        else:
            if Suffix is None: Suffix = "" 
            Name = os.path.join(Fold,File+Suffix)
        SetupOutput(Name)
        return Name
    else:
        return None
    
# Make Folder for a Root Path (FOLD/fileroot)    
def SetupOutput(RootPath):
    MakeNewDir(os.path.dirname(RootPath))
    
# MakeDir
def MakeNewDir(FOLD,verbose=False):
    if not os.path.exists(FOLD):
        os.makedirs(FOLD)
        if verbose: print("Folder created at: " + FOLD)

#Add Extension if it is missing
def FixName(Name,Ext):
    if not Ext[0] == '.':
        Ext='.'+Ext
    if not Name[-len(Ext):].lower()==Ext.lower():
        Name=Name+Ext
    return Name    
    
# GetFiles that match Name and/or Extension
def getFiles(Folder,Extension=None,PreName=None,MidName=None,PostName=None):
    
    Name="*"
    if PreName:
        Name=PreName+"*"
    if MidName:
        Name+=Midname+"*"
    if PostName:
        Name+=PostName+"."
    
    ext="*"
    if Extension:
        if Extension[0]==".": Extension=Extension[1:]
        ext=Extension
    Name+=ext    
    
    vlist = glob.glob(os.path.join(Folder, Name))
    vlist.sort()
    
    return vlist


#Rename that overwrites
def rename(src,dest):
    if os.path.isfile(dest): os.remove(dest)
    os.rename(src,dest)

#Check if the Input is newer than the Output
def IsNew(Input,Output,Force=False):
    if Force or (Output is None) or (not os.path.isfile(Output)):
        return True
    IStat = os.stat(Input).st_mtime
    OStat = os.stat(Output).st_mtime
    if IStat > OStat: return True
    return False

#Get Current Script Folder
def GetHome():
    scriptf = os.path.abspath(inspect.stack()[-1][1])
    return os.path.dirname(scriptf)



### MATPLOTLIB ###

# QuickPlot
def QPlot(x,y):
    if not type(y[0])==type([]):
        y=[y]
    
    fig = plt.figure(figsize=(8,6))
    axs  = fig.add_subplot(111)
    for yy in y:
        axs.plot(x,yy,marker='.',ls='-',label='',markersize=5)
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)

# Save figure    
def SaveMPLfig(fig,FileName=None):
    if FileName is None:
        fig.show()
        plt.waitforbuttonpress()
    else:
        FileName=FixName(FileName,"pdf")
        try:
            fig.savefig(FileName)  
        except:
            print("PDF IS OPEN!")
            ct=timeit.time.strftime("%Y%m%d%H%M%S",timeit.time.localtime())
            fig.savefig(FileName[:-4]+' (conflict_'+ct+').pdf')       

def QuickHist(X,N=50):
    X=np.array(X).flatten()
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(X,N) #Histogram
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)
            
            
### OTHER ###

# Circular Moving Average
def CircularMovingAverage(InputVector, n=3, LastIsFirst=True):
    a=InputVector
    if LastIsFirst:
        a=InputVector[:-1]
    k=int((n-1)/2)
    ret=np.concatenate((a[-k:],a,a[:k]))
    ret = np.cumsum(ret, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    out = ret[n - 1:] / n
    if LastIsFirst:
        out=np.concatenate((out,[out[0]]))
    return out

#Run a MeshLab Script
def RunMeshLabScript(Mesh_Path,Script_Path,Output_Path=None,Verbose=False):
    if Output_Path is None:
        Base,Ext = os.path.splitext(Mesh_Path)
        Output_Path = Base + "_out" + Ext
    
    SetupOutput(Output_Path)
    cmd = "meshlabserver -i " + Mesh_Path + " -o " + Output_Path + " -s " + Script_Path
    print(cmd)
    RunProgram(cmd,Verbose)
    
#Run a Shell process
def RunProgram(cmd,f_print=True):
	process = subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
	while process.poll() is None:
		if f_print:
			outl = process.stdout.readline().strip()
			print(outl)
	
	if f_print: print(process.stdout.read())



#Tic/Toc
def Tic(p=False):
    ''' Gets current time. Returns current time. p=True also prints to screen ''' 
    t=timeit.default_timer()
    if p: print("Tic: " + str(t))
    return t
    
def Toc(s,p=False):
    ''' Gets current time and subtracts input time. Returns time difference. p=True also prints to screen ''' 
    t=timeit.default_timer()
    d=t-s
    if p: print("Toc: " + str(d))
    return d
    
#Time a function
def Timeme(funct,var,NN=10,NNN=10,show=True):
    TotTime=0
    for i in xrange(NN):
        start =timeit.default_timer()
        for t in xrange(NNN):
            funct(*var)
        end =timeit.default_timer()
        TimeDiff=(end - start)/NNN*1000
        TotTime+=TimeDiff
        if show: print(str(i)+': '+str(TimeDiff))
    return TotTime/NN

        
#Get data from CSV format to numpy matrix format
def getMatfromCSV(fn):     
    data=[]
    if not fn[-4:].lower() == '.csv':
        fn=fn+".csv"
    with open(fn, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            data.append(np.array(row).astype(np.float32))
    return np.array(data)

#Apply McAuley brackets    
def macauley(M,p=True):
    if p:
        M[M<0]=0
    else:
        M[M>0]=0
    return M
    
# Invert Dictionary
def invidct(mydict):
    return {v: k for k, v in mydict.items()} 

    
#Compute eigenvalues of symetric 3x3 matrix    
def eig33s(A11,A22,A33,A12,A13,A23):

    p1 = A12**2 + A13**2 + A23**2
    if (p1 == 0): # A is diagonal.
        eig1 = A11
        eig2 = A22
        eig3 = A33
    else:
        q = (A11+A22+A33)/3
        p2 = (A11 - q)**2 + (A22 - q)**2 + (A33 - q)**2 + 2 * p1
        p = np.sqrt(p2 / 6)
        detB = (A11-q)*(A22-q)*(A33-q) + 2*A12*A13*A23 -(A11-q)*A23**2 -(A22-q)*A13**2 -(A33-q)*A12**2 
        r = detB / (2 * p**3)

        # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        # but computation error can leave it slightly outside this range.
        if (r <= -1) :
            phi = np.pi / 3
        elif (r >= 1):
            phi = 0
        else:
            phi = np.arccos(r) / 3

        # the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1 = q + 2 * p * np.cos(phi)
        eig3 = q + 2 * p * np.cos(phi + (2*np.pi/3))
        eig2 = 3 * q - eig1 - eig3     # since trace(A) = eig1 + eig2 + eig3
    return eig1,eig2,eig3    
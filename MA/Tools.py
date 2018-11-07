#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import inspect
import shutil
import timeit
import time
import glob
import csv
import os


### File and Folder Manipulation ###

def DeleteFolderTree(Folder):
    ''' Deletes Folder Tree '''
    if os.path.exists(Folder):
        #TODO: add exception handling for read-only files
        shutil.rmtree(Folder)

def MakeRoot(Root=None,Suffix=None):
    ''' Combine Root and Suffix for a path. It's robust '''
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

def SetupOutput(RootPath):
    '''Make Folder for a Root Path (FOLD/fileroot)'''
    MakeNewDir(os.path.dirname(RootPath))
    
def MakeNewDir(FOLD,verbose=False):
    '''Make New Folder if it does not already exist'''
    if not os.path.exists(FOLD):
        os.makedirs(FOLD)
        if verbose: print("Folder created at: " + FOLD)

def FixName(Name,Ext):
    '''Add Extension if it is missing'''
    if not Ext[0] == '.':
        Ext='.'+Ext
    if not Name[-len(Ext):].lower()==Ext.lower():
        Name=Name+Ext
    return Name    
    
# 
def getFiles(Folder,Extension=None,PreName=None,MidName=None,PostName=None):
    '''Search folder and returns list of files that match Name and/or Extension. Equivalent to
    doing Folder/PreName*MidName*PostName.Extension

    Input:
        Folder -- Folder to Search
        optional -- Extension,PreName,MidName,PostName
    '''
    
    Name="*"
    if PreName:
        Name=PreName+"*"
    if MidName:
        Name+=MidName+"*"
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

def rename(src,dest):
    '''Rename that overwrites'''
    if os.path.isfile(dest): os.remove(dest)
    os.rename(src,dest)

def AnyIsNew(Inputs,Output,Force=False):
    '''Check if any of the Inputs is newer than the Output'''
    if Force or (Output is None) or (not os.path.isfile(Output)):
        return True
    for Input in Inputs:
        if IsNew(Input,Output,Force=Force): return True
    return False


def IsNew(Input,Output,Force=False):
    '''Check if the Input is newer than the Output'''
    if Force or (Output is None) or (not os.path.isfile(Output)):
        return True
    IStat = os.stat(Input).st_mtime
    OStat = os.stat(Output).st_mtime
    if IStat > OStat: return True
    return False

def GetHome():
    '''Get Current Script Folder'''
    scriptf = os.path.abspath(inspect.stack()[-1][1])
    return os.path.dirname(scriptf)

def GetBinRepr(InpFile,OutFile=None):
    '''Gets the binary representation of a file as text'''
    with open(InpFile,"rb") as fp: A=repr(fp.read())
    if OutFile is not None:
        with open(OutFile,"w+") as fp: fp.write(A)
    else:
        print(A)

### MATPLOTLIB ###

def QPlot(x,y):
    ''' Quickly plot line(s).

    x - array with xvalues
    y - array or list of arrays with y values.
    '''
    if not type(y[0])==type([]):
        y=[y]
    
    fig = plt.figure(figsize=(8,6))
    axs  = fig.add_subplot(111)
    for yy in y:
        axs.plot(x,yy,marker='.',ls='-',label='',markersize=5)
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)


def QMPlot(Xa,Ya,marker=""):
    ''' Quickly plot multiple lines.
    Xa - list of arrays with x values
    Ya - list of arrays with y values
    marker[""] - change to add a marker to the plots (e.g. marker="o")
    '''
    fig = plt.figure(figsize=(8,6))
    axs  = fig.add_subplot(111)
    linelist=["-",":","-."]
    for x,y in zip(Xa,Ya):
        axs.plot(x,y,marker=marker,ls=linelist[0],label='',markersize=5)
        linelist.insert(0, linelist.pop())
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)
 
def SaveMPLfig(fig,FileName=None):
    ''' Show or robust save a Matplotlib figure. Will generate a new pdf file if the 
    current one is open instead of failing.

    fig - fig object from matplotlib
    FileName[None] - Name of file to save. If file is open, instead of failing it 
                     saves the file in a new filename with the date.
    '''
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
    '''Quicky make a histogram of X.
    N[50] - number of bins for the histogram.''' 
    X=np.array(X).flatten()
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(X,N) #Histogram
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)
            
            
### OTHER ###

def CircularMovingAverage(InputVector, n=3, LastIsFirst=True):
    ''' Compute a Circular Moving Average, i.e. first and last elements influence eachother.
    
    InputVector - Array for which the moving average will be computed.
    n[3] - size of the moving average
    LastIsFirst[True] - Set to true is first and last values are the same point in the circle (e.g. 0 degrees and 360 degrees)
    '''
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

def RunMeshLabScript(Mesh_Path,Script_Path,Output_Path=None,Verbose=False,MeshLabPath=None):
    '''Run a MeshLab Script.
    
    Mesh_Path - Path to mesh
    Script_Path - Path to script
    Output_Path[None] - Path to output file. Default of None will generate an output path automatically.
    Verbose[False] - T/F print the output of running the script.
    MeshLabPath[None] - path of Meshlab. Default of None will try to run script assuming meshlab is in the PATH environment variable'''

    if Output_Path is None: # Create an automatic output file in user does not provide one
        Base,Ext = os.path.splitext(Mesh_Path) # Get file name and extensions seperately
        Output_Path = Base + "_out" + Ext 
    
    SetupOutput(Output_Path) 
    if MeshLabPath is None: # If path of Meshlab not given, hope that meshlabserver path is in the PATH environment variable
        prog = "meshlabserver"
    else:
        prog = '"'+os.path.join(MeshLabPath,"meshlabserver")+'"'
        

    cmd = prog+' -i '+ Mesh_Path + " -o " + Output_Path + " -s " + Script_Path
    print("  "+cmd)
    RunProgram(cmd,Verbose)
    print("  Meshlab Script Finished")

#Run a Shell process
def RunProgram(cmd,f_print=True):
    ''' Run comand in command line.

    cmd - String with the command (example: git pull)
    f_print[True] - Boolean that will print the output of running the command''' 
    process = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    while process.poll() is None:
        outl = process.stdout.readline().strip()
        if f_print:
            print(outl)
    outl=process.stdout.read()
    if f_print: print(outl)



#Tic/Toc
def Tic(p=False):
    ''' Gets and returns current time.
    p[True] - Prints current time to screen. ''' 
    t=timeit.default_timer()
    if p: print("Tic: " + str(t))
    return t
    
def Toc(s,p=False):
    ''' Gets current time and subtracts input time. Returns time difference.
    p[True] - Prints current time to screen. ''' 
    t=timeit.default_timer()
    d=t-s
    if p: print("Toc: " + str(d))
    return d
    
#Time a function
def Timeme(funct,var,NN=10,NNN=10,show=True):
    ''' Measure the computational time of a function. Returns average time across NN runs of NNN trials.

    funct - Function pointer to the function to be measured.
    var - List/Tuple with variables that are needed as inputs for the function.
    NN - Integer with the number of tests runs.
    NNN - Number of trials for each of the test runs. Test run result is the average across NNN times.
    show[True] - T/F Show the results on the screen.
    '''
    TotTime=0
    for i in range(NN):
        start =timeit.default_timer()
        for _ in range(NNN):
            funct(*var)
        end =timeit.default_timer()
        TimeDiff=(end - start)/NNN*1000
        TotTime+=TimeDiff
        if show: print(str(i)+': '+str(TimeDiff))
    return TotTime/NN

def Pause(seconds):
    '''Pause the simulation for X number of seconds'''
    time.sleep(seconds)
    
def getMatfromCSV(fn):
    '''Get data from CSV format to numpy matrix format. Probably better to use pandas'''
    data=[]
    if not fn[-4:].lower() == '.csv':
        fn=fn+".csv"
    with open(fn, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            data.append(np.array(row).astype(np.float32))
    return np.array(data)

   
def macauley(M,p=True):
    '''Apply McAuley brackets. f(x)=x if x>0 else 0.

    p[True] - T/F Use positive x. If false f(x)=x if x<0 else 0.'''
    if p:
        M[M<0]=0
    else:
        M[M>0]=0
    return M
    
def invidct(mydict):
    '''Invert dictionary making the values keys and the keys values'''
    return {v: k for k, v in list(mydict.items())} 

    
def eig33s(A11,A22,A33,A12,A13,A23):
    '''Compute eigenvalues of symetric 3x3 matrix'''

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

def ConvertToStructArray(array):
    ''' Converts a numpy array into a structured array.

    Converts a (m x n) array into an array of length m where each entry is
    a tuple of length n with the contents of line m of the original array.

    The output is a structured array where each column is called "fx" where
    x is an integer starting at 0 and ending at (n-1), for each of the columns
    of the original array, respectively.

    Input: array-like object
    Output: Numpy structured array
    '''

    array=np.array(array)
    dty=array.dtype.name
    if 'str' in dty:
        raise ValueError("Undifined array data type (" + dty + "). Check to make sure all entries are of the same type and not strings")

    new_dtype = dty
    for _ in range(array.shape[1]-1):
        new_dtype += "," + dty

    return array.view(dtype=new_dtype).copy()




from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import bytes
from builtins import str
import numpy as np
import os
import MA.ImageProcessing as MAIP
import MA.Tools as MATL   
from PIL import Image
    
'''
This module was originally created to convert a CSV file into a 3D file.
It now also converts from a tiff image file and any 2D matrix.

The precess is to assume that the 2D matrix is a representation of a 3D surface,
where the x and y coordinates are obtained from the position of each element of the
grid of numbers (column number for x and row number for y), potentially scaled by
a horizontal scale factor. The z coordinate is assumed to be the value of each element,
potentially scaled by a vertical scale factor.

To convert a grayscale tiff image to a ply file do the following:
    import MA.CSV3D as MA3D
    MA3D.Make3DSurfaceFromHeightMapTiff(TiffFile,OFile=output_ply_file,NoZeros=True/False)
'''


def MakeNodes(npdata,Scale=(1.0,1.0,1.0)):
    ''' Internal function to generate each node of the mesh'''
    Nr,Nc= np.shape(npdata)
    Nodes=np.zeros(Nr*Nc,dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
    x = np.linspace(0, (Nc-1), Nc, dtype=np.float32)
    y = np.linspace((Nr-1),0 , Nr, dtype=np.float32)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    Nodes['x']=xv.flatten()*Scale[0]
    Nodes['y']=yv.flatten()*Scale[1]
    Nodes['z']=npdata.flatten()*Scale[2]         
    return Nodes

def MakeColors(Mpix):
    ''' Internal function to generate a color for each node of the mesh'''
    Nr,Nc = np.shape(Mpix)
    Colors=np.zeros(Nr,dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('a', np.uint8)])
    Colors['r']=Mpix[:,0]
    Colors['g']=Mpix[:,1]
    Colors['b']=Mpix[:,2]
    Colors['a']=Mpix[:,3] 
    return Colors
   
def MakeFaces(Nr,Nc):
    ''' Internal function to generate each element of the mesh'''
    out = np.empty((Nr-1,Nc-1,2,3),dtype=int)
    r = np.arange(Nr*Nc).reshape(Nr,Nc)
    l1=r[:-1,:-1]
    l2=r[:-1,1:]
    l3=r[1:,:-1]
    l4=r[1:,1:]
    out[:,:, 0,0] = l2
    out[:,:, 0,1] = l1
    out[:,:, 0,2] = l3
    out[:,:, 1,0] = l4
    out[:,:, 1,1] = l2
    out[:,:, 1,2] = l3
    out.shape =(-1,3)
    return out

def ExportPlyBinary(Nodes,file,Faces=None,Colors=None):
    ''' Internal function to export the mesh into a ply file'''
    LN=len(Nodes)
    LF=len(Faces)
    if Faces is None:
        LF=0
    else:
        LF=len(Faces)
    if Colors is None:
        cprop=""
    else:
        cprop= \
        "property uchar red\n" \
        "property uchar green\n" \
        "property uchar blue\n" \
        "property uchar alpha\n"

    header= \
    "ply\n" \
    "format binary_little_endian 1.0\n" \
    "element vertex "+str(LN)+"\n" \
    "property float x\n" \
    "property float y\n" \
    "property float z\n" \
    + cprop + \
    "element face "+str(LF)+"\n" \
    "property list uchar int vertex_indices\n" \
    "end_header\n"
    header = bytes(header,'utf-8')
    
    dtype_vertex = [('vertex', '<f4', (3))]
    if Colors is not None: dtype_vertex.append(('rgba',   '<u1', (4)))
    vertex = np.empty(LN, dtype=dtype_vertex)
    vertex['vertex']=np.stack((Nodes['x'],Nodes['y'],Nodes['z']),axis=-1)
    if Colors is not None: vertex['rgba']=np.stack((Colors['r'],Colors['g'],Colors['b'],Colors['a']),axis=-1)
    # vertex=Nodes
    
    dtype_face = [('count', '<u1'),('index', '<i4', (3))]
    faces = np.empty(LF, dtype=dtype_face)
    faces['count'] = 3
    faces['index'] = Faces

    with open(file, 'wb') as fp:
        fp.write(header)
        fp.write(vertex.tostring())
        fp.write(faces.tostring())

def MakeCSVMesh(CSVfile,verbose=True,**kwargs):
    ''' Generates a ply mesh from a csv file '''
    if verbose: print("*Making CSV Mesh")
    if verbose: print("  - Reading .csv")
    npdata=MATL.getMatfromCSV(CSVfile)
    return MakeMPixMesh(npdata,verbose,**kwargs)
    
def MakeMPixMesh(npdata,verbose=True,Scale=(1.0,1.0,1.0),ColFile=None,ColMat=None,Expfile=None,NoZeros=False):
    ''' Generates a ply mesh from a 2D array
    
    verbose[True] - If True it prints out the progress of the algorithm
    Scale[(1.0,1.0,1.0)] - Scaling factors in x,y and z directions. Final coordinates = scale * input coordinates (eg. pixel)
    ColFile[None] - Image file with the colors to export in the ply file
    ColMat[None] - Matrix with colors to export in the ply file (is overriden by ColFile)
    Expfile[None] - Path to output ply file
    NoZeros[False] - Do not generate mesh when pixel/element value is equal to zero.
    '''
    if ColFile is not None:
        if verbose: print("  - Getting Image")
        Mpix=MAIP.GetRGBAImageMatrix(ColFile,Silent=not verbose).reshape(-1,4)
        Colors=MakeColors(Mpix)
    elif ColMat is not None:
        if verbose: print("  - Getting Image")
        Colors=MakeColors(ColMat)
    else:
        Colors=None
    
    if verbose: print("  - Generating Nodes")
    Nodes=MakeNodes(npdata,Scale)
    if verbose: print("  - Generating Triangles")
    Nr,Nc= np.shape(npdata)
    Faces=MakeFaces(Nr,Nc)

    if NoZeros:
        NodesNotZero = Nodes['z']!=0.0
        NN=len(Nodes)
        NodesToKeep = np.arange(NN)[NodesNotZero]
        Mapping = np.zeros(NN)
        Mapping[NodesToKeep]=np.arange(len(NodesToKeep))
        FacesToKeep = np.apply_along_axis(all, 1, NodesNotZero[Faces])
        Nodes = Nodes[NodesToKeep]
        Faces = Mapping[Faces[FacesToKeep]]

    if Expfile is not None:
        if verbose: print("  - Exporting Geometry")
        ExportPlyBinary(Nodes,Expfile,Faces,Colors)
    return Nodes,Faces,Colors    
    
def Make3DSurfaceFromHeightMapTiff(File,OFile=None,NoZeros=False):
    ''' Generates a ply mesh from a tiff file. Scale is extracted from tiff embedded information.
    
    File - Tiff file with the height information
    OFile[None] - Path to output ply file
    NoZeros[False] - Do not generate mesh when pixel/element value is equal to zero.
    '''
    name,_ = os.path.splitext(File)
    if OFile is None:
        outfile=name+"_out.ply"
    else:
        outfile=MATL.FixName(OFile,"ply")

    Mpix,res = MAIP.GetImageMatrix(File,Silent=True,GetTiffRes=True)

    return MakeMPixMesh(Mpix,Scale=(1/res[0],1/res[1],1.0),Expfile=outfile,NoZeros=NoZeros)   
    
    
if __name__ == "__main__":
    print("=Running Test")
    fn="Tests/CSV3D/test"
    MakeCSVMesh(fn+".csv",ColFile=fn+".png",Expfile=fn+".ply")
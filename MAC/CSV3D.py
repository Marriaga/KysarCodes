from __future__ import division
import numpy as np
import os
import MAPyLibs.ImageProcessing as MAIP
import MAPyLibs.Tools as MATL   
from PIL import Image
    
def MakeNodes(npdata,Scale=(1.0,1.0,1.0)):
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
    Nr,Nc = np.shape(Mpix)
    Colors=np.zeros(Nr,dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8), ('a', np.uint8)])
    Colors['r']=Mpix[:,0]
    Colors['g']=Mpix[:,1]
    Colors['b']=Mpix[:,2]
    Colors['a']=Mpix[:,3] 
    return Colors
   
def MakeFaces(Nr,Nc):
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

def MakeCSVMesh(CSVfile,**kwargs):
    print "*Making CSV Mesh"
    print "  - Reading .csv"
    npdata=MATL.getMatfromCSV(CSVfile)
    return MakeMPixMesh(npdata,**kwargs)
    
def MakeMPixMesh(npdata,Scale=(1.0,1.0,1.0),ColFile=None,ColMat=None,Expfile=None):
    if ColFile is not None:
        print "  - Getting Image"
        Mpix=MAIP.GetRGBAImageMatrix(ColFile).reshape(-1,4)
        Colors=MakeColors(Mpix)
    elif ColMat is not None:
        print "  - Getting Image"
        Colors=MakeColors(ColMat)
    else:
        Colors=None
    
    print "  - Generating Nodes"
    Nodes=MakeNodes(npdata,Scale)
    print "  - Generating Triangles"
    Nr,Nc= np.shape(npdata)
    Faces=MakeFaces(Nr,Nc)
    if Expfile is not None:
        print "  - Exporting Geometry"
        ExportPlyBinary(Nodes,Expfile,Faces,Colors)
    return Nodes,Faces,Colors    
    
def Make3DSurfaceFromHeightMapTiff(File):
    name,ext = os.path.splitext(File)
    outfile=name+"_out.ply"
    myimg=Image.open(File)
    xres,yres = myimg.info['resolution']
    Mpix = np.copy(np.asarray(myimg))
    MakeMPixMesh(Mpix,Scale=(1/xres,1/yres,1.0),Expfile=outfile)   
    
    
    
    
if __name__ == "__main__":
    print "=Running Test"
    fn="Tests/CSV3D/test"
    MakeCSVMesh(fn+".csv",ColFile=fn+".png",Expfile=fn+".ply")
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
import numpy as np
import numpy.lib.recfunctions as nprec
import sys
import os
import MA.Tools as MATL
import MA.ImageProcessing as MAIP
import xml.dom.minidom
import numba as nb


'''
MeshIO Module
=============

This module deals with importing/exporting and processing meshes.

There are two types of classes:

1) Mesh classes - These have the necessary data structures and functions to
   process the mesh information, including computation of normals and curvatures:
    - CNodes
    - CElements
    - MyMesh

2) I/O classes - These convert to these formats from the data in the Mesh classes.
   currently PLYIO also reads ply data into thr Mesh classes format:
    - BaseIO
    - PLYIO
    - VTUIO
    - FEBioIO

Typical usage:

import MA.MeshIO as MAMIO

# Import a mesh data from a ply file

Ply = MAMIO.PLYIO()
Ply.LoadFile(plyFile_Smooth)        # Load Ply File
Nodes,Elems = Ply.ExportMesh()      # Get Data in mesh classes format
MyM = MAMIO.MyMesh(Nodes,Elems)     # Create Mymesh object

# Process the mesh data

MyM.ComputeNormals()                # Compute Normals
MyM.ComputeCurvatures()             # Compute Curvatures
MyM.SmoothCurvatures(SmoothN)       # Smooth Curvatures 
MyM.ComputeCurvaturePrincipalDirections() # Compute Principal Directions
MyM.ReadjustMinCurvDirections() # Readjust Min Curvature Directions
MyM.Nodes.AppendArray(MyM.NNorms,'n') # Add Normal Vector to the nodal data

# Output the mesh data into a ply format

Ply = MAMIO.PLYIO()
Ply.ImportMesh(MyM.Nodes,MyM.Elems)
Ply.SaveFile(path/to/ouputfile.ply)

# (Alternatively) Output the mesh data into a vtu format

Vtu = MAMIO.VTUIO()
Vtu.VecLabs.append(["normal",["nx","ny","nz"]])
Vtu.ImportMesh(MyM.Nodes,MyM.Elems)
Vtu.SaveFile(path/to/ouputfile.vtu)
'''


# Get Endienness of System
SYS_ENDI = {'little': '<','big': '>'}[sys.byteorder]

# Dictionary for ply->numpy endienness
ENDI_CONVERSION = {
'ascii': '=',
'binary_little_endian': '<',
'binary_big_endian': '>'}
# Dictionary for numpy->ply endienness
INV_ENDI_CONVERSION = MATL.invidct(ENDI_CONVERSION)


def GetEndinp(val):
    '''Get Endi in numpy format (from ply)'''
    return ENDI_CONVERSION[val]    

def GetEndiply(val):
    '''Get Endi in ply format (from numpy)'''
    return INV_ENDI_CONVERSION[val]
    
# Dictionary for ply->numpy data types  
TYPE_CONVERSION ={
'char'  :'i1',
'uchar' :'u1',
'short' :'i2',
'ushort':'u2',
'int'   :'i4',
'uint'  :'u4',
'float' :'f4',
'double':'f8'}
# Dictionary for numpy->ply data types  
INV_TYPE_CONVERSION=MATL.invidct(TYPE_CONVERSION)

def ply2numpy(plytype,endi):
    '''Get datatype in numpy format (from ply)
    
    endi: < - little; > - Big
    '''
    if endi == "=": endi = SYS_ENDI
    return endi + TYPE_CONVERSION[plytype]
    
def numpy2ply(nptype):
    '''Get datatype in ply format (from numpy)'''
    if nptype[0] in "=<>": nptype = nptype[1:]
    return INV_TYPE_CONVERSION[nptype]

# Class for collection of properties    
class MyPropTypes(object):
    def __init__(self):
        ''' Class for collection of properties. This class keeps track of properties and
        fields for nodes and elements. It contains default properties line node number,
        x,y,z positions, rgb values; and also allows the addition of arbitrary new fields.

        Note: Generally this class should not be used directly.
        '''
        self.AllTypes=dict()
        self.Ltype=[]
        
        #Defaults
        self.AllTypes["N"]=SYS_ENDI+'i4'
        self.AllTypes["x"]=SYS_ENDI+'f8'
        self.AllTypes["y"]=SYS_ENDI+'f8'
        self.AllTypes["z"]=SYS_ENDI+'f8'
        self.AllTypes["r"]=SYS_ENDI+'u1'
        self.AllTypes["g"]=SYS_ENDI+'u1'
        self.AllTypes["b"]=SYS_ENDI+'u1'
        self.AllTypes["a"]=SYS_ENDI+'u1'
        
        self.AllTypes["p1"]=SYS_ENDI+'i4'
        self.AllTypes["p2"]=SYS_ENDI+'i4'
        self.AllTypes["p3"]=SYS_ENDI+'i4'
        
    def AddBaseType(self,typlbl,type=None):
        '''Changes or adds a type in the available types'''
        if type is None:
            type='f8'
            print("Warning: Added label "+typlbl+" with no associated type. 'double' assumed")
        self.AllTypes[typlbl] = type
    
    def AddTypeToList(self,typlbl,type=None):
        '''Adds a type to the type list'''
        if (typlbl not in self.AllTypes) or (type is not None):
            self.AddBaseType(typlbl,type)
        
        if typlbl not in self.Ltype:
            self.Ltype.append(typlbl)
            
    def GetType(self,typlbl):
        '''Get the type of a field with a particular label'''
        if typlbl not in self.AllTypes:
            self.AddBaseType(typlbl)
        return self.AllTypes[typlbl]
    
    def NormEndi(self,Endi):
        '''Normalize Endienness'''
        for typlbl in self.Ltype:
            typtyp = self.GetType(typlbl)
            if typtyp[0] in "=<>|":
                typtyp = typtyp[1:]
            typtyp=Endi+typtyp
            self.AddBaseType(typlbl,typtyp)
    
    def GetDtype(self):
        '''Get Numpy dtype data'''       
        ListDTypes=[]
        for typlbl in self.Ltype:
            ListDTypes.append((typlbl,self.GetType(typlbl)))
        return ListDTypes
    
    def GetNProp(self):
        '''Get number of properties present'''
        return len(self.Ltype)
        
    def MakeListType(self,Types,append=False):
        '''Generates the list of types.
        
        Types - types to set/append. If list, then list is labels and
                data type will be inferred automatically. If numpy dtype,
                then labels and types are extracted. If MyPropTypes object,
                copy the list from the MyPropTypes object.
        append[False] - If True, append to current list, otherwise replaces.
        '''

        if not append:
            self.Ltype = []
        if type(Types) is np.dtype:
            for lbl,typ in Types.dtype.descr:
                self.AddTypeToList(lbl,typ)
        elif type(Types) is type(self):
            self.AllTypes=Types.AllTypes
            self.Ltype=Types.Ltype
        else:
            for lbl in Types:
                self.AddTypeToList(lbl)
   
# Class for Nodes
class CNodes(object):
    def __init__(self,Matrix=None,**kwargs):
        '''This class manages nodal data. It imports coordinates in
        a matrix form and allows for addition of new fields.

        Matrix[None] - Array of (N),x,y,(z) values. It will be processed
        and imported into the class.
        **kwargs - Additional arguments that will be passed to function
        that imports the Matrix called PutMatrix().
        '''
        self.is_sequencial=False # Nodes are sequencial in numbering
        self.si=0      # Starting index
        self.Mat=[]    # Numpy array with values
        self.PropTypes = MyPropTypes() # Data types
        
        if Matrix is not None:
            self.PutMatrix(Matrix,**kwargs)
        
    def GetNNodes(self):
        '''Returns number of nodes'''
        return len(self.Mat)
    
    @staticmethod
    def _getArrayLabel(root,nf,i,UseLetters=False):
        '''Internal function to make array labels from root label.

        root - string for the name of the field
        nf - size of the dimension (x,y,z has nf=3)
        i - current dimension index (0 -> x)
        UseLetters - use x,y,z instead of 0,1,2

        Examples:
            _getArrayLabel('disp',3,2,UseLetters=True) ->'dispz'
            _getArrayLabel('v',3,2) ->'v2'
        '''
        letter=['x','y','z']
        if nf<=3 and UseLetters:
            lbl=root+letter[i]
        else:
            lbl=root+str(i)
        return lbl

    def AppendArray(self,myarray,root,UseLetters=True):
        '''Appends an array to the nodes as several fields based
        on the root label. dim(myarray) = NumNodes x M (x N)-> saved as
        MxN fields each labeled "rootij", with i,j in [x,y,z] or [0,1,2,3,...]

        myarray - array to add with dimensions NumNodes x M (x N)
        root - string for the name of the field
        UseLetters - use x,y,z instead of 0,1,2
        '''
        shp=np.shape(myarray)
        if len(shp)>3:
            raise ValueError("ERROR: Export bigger than matrix is not implemented.")
        elif len(shp)==3:
            nf = shp[1]
            for i in range(nf):
                lbl=CNodes._getArrayLabel(root,nf,i,UseLetters=UseLetters)
                self.AppendVector(myarray[:,i,:],lbl,UseLetters=UseLetters)
        else:
            self.AppendVector(myarray,root,UseLetters=UseLetters)


    def AppendVector(self,myarray,root,UseLetters=True):
        '''Appends a vector to the nodes as several fields based
        on the root label. dim(myarray) = NumNodes x M -> saved as
        M fields each labeled "rooti", with i in [x,y,z] or [0,1,2,3,...]

        myarray - array to add with dimensions NumNodes x M
        root - string for the name of the field
        UseLetters - use x,y,z instead of 0,1,2
        '''
        shp=np.shape(myarray)
        nf = shp[1]
        for i in range(nf):
            lbl=CNodes._getArrayLabel(root,nf,i,UseLetters=UseLetters)
            self.AppendField(myarray[:,i],lbl)

    def AppendField(self,myarray,lbl,atype=None):
        '''Appends a field to the nodes.

        myarray - array to add with length = NumNodes
        lbl - string for the name of the field
        atype[None] - datatype of values. 'None' sets datatype from numpy.
        '''    
        if atype is not None:
            myarray=np.array(myarray,dtype=atype)
        else:
            myarray=np.array(myarray)
            atype=myarray.dtype.descr[0][1]
        
        if lbl not in self.PropTypes.Ltype:
            self.Mat=nprec.append_fields(self.Mat,lbl,myarray)
            self.PropTypes.AddTypeToList(lbl,atype)
        else:
            self.Mat[lbl]=myarray
    
    def PutMatrix(self,Matrix,Types=None,isnumbered=False):
        '''Takes the Matrix array, processes it and creates the Numpy
        array with the correct data type.

        Matrix - list/array in raw format. Typically NumNodes x 3 (x,y,z)
        Types[None] - set data types. 'None' infers from dimensions and 'isnumbered'
        isnumbered[False] - set to True if first column has node numbers.
        '''

        Matrix=np.array(Matrix)
        try:
            NN,NF = np.shape(Matrix)
            Matrix=MATL.ConvertToStructArray(Matrix)
        except ValueError:
            NN = len(Matrix)
            NF = len(Matrix.dtype.names)

        # Check data
        if (NF == 1) or (NF == 2 and isnumbered): raise ValueError("1D arrays do not make sense")        
        # Set dtype labels
        if Types is not None:
            self.PropTypes.MakeListType(Types)
            typelabs = self.PropTypes.Ltype
        elif NF <= 4:
            typelabs=['x','y','z']
        else:
            typelabs=Matrix.dtype.names
        
        # Add N dtype label         
        if not isnumbered and 'N' in typelabs:
            raise ValueError("'N' in labels. Set 'isnumbered' flag to True")
        elif not 'N' in typelabs:
            typelabs = ['N'] + typelabs
        
        # Set Dtype
        self.PropTypes.MakeListType(typelabs)
        
        # Add numbering of nodes if needed
        if not isnumbered:
            Numbering = np.arange(NN).view(dtype=[('N', self.PropTypes.GetType('N'))])
            Matrix=nprec.merge_arrays((Numbering,Matrix), flatten = True)
            
        # Add z=0 for 2D Data    
        if (NF == 2) or (NF == 3 and isnumbered): #Assumed 2D Structure
            ZeroZ = np.zeros(NN).view(dtype=[('z', self.PropTypes.GetType('z'))])
            Matrix=nprec.merge_arrays((Matrix,ZeroZ), flatten = True)
        
        # Save Matrix
        Matrix=Matrix.astype(self.PropTypes.GetDtype())
        self.Mat = np.sort(Matrix,order='N')

        # Check ordering and starting item
        MN = self.Mat['N']
        if MN[-1]-MN[0]+1 == len(MN):
            self.is_sequencial=True
            self.si=MN[0]
        
    def RotateAndTranslate(self,R,T):
        ''' Apply rotation R and translation T to mesh.

        R - List/Tuple with Euler angles (phi,theta,psi)
        T - List/Tuple with translation vector (dx,dy,dz)

        TODO: If would be interesting to add the possibility
        to also rotate vector and tensor fields.
        '''
        x = self.Mat[['x','y','z']]
        Coords = x.view((x.dtype[0], len(x.dtype.names))).copy()
        RCT = MAIP.CoordsObj.ApplyRotationAndTranslation(Coords,R,T,pavg=0)

        self.Mat['x'] =  RCT[:,0].copy()
        self.Mat['y'] =  RCT[:,1].copy()
        self.Mat['z'] =  RCT[:,2].copy()

    def GetNumpyPoints(self,Nlist):
        ''' Get list of numpy arrays, where each numpy array
        contains the 3D coordinates of each node in Nlist.

        Nlist - List of node numbers to get the coordinates
        '''
        if type(Nlist) is int:
            Nlist = [Nlist]
        return [np.array([P['x'],P['y'],P['z']]) for P in [self.Mat[n] for n in Nlist]]

# Class for Elements
class CElements(object):
    def __init__(self,Matrix=None,**kwargs):
        '''This class manages element data. It imports the list of nodes
        of each element in a matrix form.

        Matrix[None] - Array of (N),p1,p2,p3 values. It will be processed
        and imported into the class.
        **kwargs - Additional arguments that will be passed to function
        that imports the Matrix called PutMatrix().
        '''
        self.is_sequencial=False # Nodes are sequencial in numbering
        self.si=0      # Starting index
        self.Mat=[]    # Numpy array with values
        self.PropTypes = MyPropTypes() # Data types
        
        if Matrix is not None:
            self.PutMatrix(Matrix,**kwargs)
        
    def GetNElems(self):
        '''Returns number of elements'''
        return len(self.Mat)
        
    def PutMatrix(self,Matrix,Types=None,isnumbered=False):
        '''Takes the Matrix array, processes it and creates the Numpy
        array with the correct data type.

        Matrix - list/array in raw format. Typically NumElements x 3 (p1,p2,p3)
        Types[None] - set data types. 'None' infers from dimensions and 'isnumbered'
        isnumbered[False] - set to True if first column has node numbers.
        '''
        Matrix=np.array(Matrix)
        try:
            NN,NF = np.shape(Matrix)
            Matrix=MATL.ConvertToStructArray(Matrix)
        except ValueError:
            NN = len(Matrix)
            NF = len(Matrix.dtype.names)
        
        # Check data
        if (NF < 3) or (NF == 3 and isnumbered): raise ValueError("Number of points < 2 does not make sense")

        # Set dtype labels
        if Types is not None:
            self.PropTypes.MakeListType(Types)
            typelabs = self.PropTypes.Ltype
        elif (NF == 3) or (NF == 4 and isnumbered):
            typelabs=['p1','p2','p3']
        else:
            typelabs=Matrix.dtype.names        
        
        # Add N dtype label         
        if not isnumbered and 'N' in typelabs:
            raise ValueError("'N' in labels. Set 'isnumbered' flag to True")
        elif not 'N' in typelabs:
            typelabs = ['N'] + typelabs
        
        # Set Dtype
        self.PropTypes.MakeListType(typelabs)

        # Add numbering of nodes if needed
        if not isnumbered:
            Numbering = np.arange(NN).view(dtype=[('N', self.PropTypes.GetType('N'))])
            Matrix=nprec.merge_arrays((Numbering,Matrix), flatten = True)
        
        # Save Matrix
        Matrix=Matrix.astype(self.PropTypes.GetDtype())
        self.Mat = np.sort(Matrix,order='N')
        
        # Check ordering and starting item
        MN = self.Mat['N']
        if MN[-1]-MN[0]+1 == len(MN):
            self.is_sequencial=True
            self.si=MN[0]

# Base class for I/O
class BaseIO(object):
    def __init__(self):
        '''This is the base class for I/O. It defines the basic
        constitutints of a I/O class, including functions to
        import and export the mesh information.
        
        TODO: It would make sense to do a refactoring of the I/O classes
        such that they would only depend on a MyMesh object as opposed
        to both a CNodes object and a CElements objects
        '''
        self.PropTypes=MyPropTypes()
        self.EPT=MyPropTypes()
        self.Endi = SYS_ENDI
        self.NNodes = -1
        self.NElem = -1
        self.Nodes = []
        self.Elems = []

    def NormEndi(self):
        '''Reformat data to match Endi'''
        self.PropTypes.NormEndi(self.Endi)
        self.Nodes.Mat=self.Nodes.Mat.astype(self.PropTypes.GetDtype())
        self.EPT.NormEndi(self.Endi)
        self.Elems.Mat=(self.Elems.Mat.astype(self.EPT.GetDtype()))
        
    ## -- DATA INPUT/OUTPUT OPERATIONS -- ##
    
    def ExportMesh(self):
        '''Get CNodes and CElements objects'''
        return self.Nodes, self.Elems
        
    def ImportMesh(self,Nodes,Elems):
        '''Set CNodes and CElements objects'''
        self.Nodes = Nodes
        self.Elems = Elems
        self.RefreshNodes()
        self.RefreshElems()
        self.MeshClass = MyMesh(self.Nodes,self.Elems)
        
    def RefreshNodes(self):
        '''Refresh Local Properties for Nodes'''
        self.PropTypes = self.Nodes.PropTypes
        self.NNodes = self.Nodes.GetNNodes()
        
    def RefreshElems(self):
        '''Refresh Local Properties for Elemens'''
        self.EPT = self.Elems.PropTypes
        self.NElem = self.Elems.GetNElems()

    def SetNodes(self,Matrix,**kwargs):
        '''Set CNodes object'''
        self.Nodes = CNodes(Matrix,**kwargs)
        self.RefreshNodes()
        
    def SetElems(self,Matrix,**kwargs):
        '''Set CElements object'''
        self.Elems = CElements(Matrix)
        self.RefreshElems()
            
# Class for PLY I/O
class PLYIO(BaseIO):
    def __init__(self,Load_File=None):
        ''' Class to do I/O with ply format. So far it is the 
        only format that allows input. 

        Load_File[None] - the path to ply file to import (if not None)
        '''

        BaseIO.__init__(self)
        if Load_File is not None:
            self.LoadFile(Load_File)
    
    ## -- READING PLY OPERATIONS -- ##
    
    def LoadFile(self,plyfile):
        '''Load a ply file'''
        with open(plyfile, 'rb') as fp:
            self.ReadHeader(fp)
            if self.Endi == "=":
                self.ReadNodesASCII(fp)
                self.ReadElemsASCII(fp)
            else:
                self.ReadNodesBinary(fp)
                self.ReadElemsBinary(fp)
                
    def ReadHeader(self,fp):
        '''Internal function to read the header of a ply file'''
        stage=0
        while True:
            lvs=fp.readline().decode().rstrip().split()
            if lvs[0]=="end_header":
                break
            
            elif lvs[0]=="format":
                self.Endi = GetEndinp(lvs[1])
            
            elif lvs[0]=="element":
                if lvs[1]=="vertex":
                    self.NNodes = int(lvs[2])
                    stage = 1
                elif lvs[1]=="face":
                    self.NElem = int(lvs[2])
                    stage = 2
            
            elif lvs[0]=="property":
                if stage == 1:  # Collect Node Properties 
                    self.PropTypes.AddTypeToList(lvs[2],ply2numpy(lvs[1],self.Endi))
                elif stage == 2: #TODO: implement colection of element properties
                    pass
                      
    def ReadNodesASCII(self,fp):
        '''Internal function to read the nodes of a ply file in ASCII format'''
        Nodes = np.empty(self.NNodes,dtype=np.dtype(self.PropTypes.GetDtype()))
        NP = self.PropTypes.GetNProp()
        for i in range(self.NNodes):
            linelist=fp.readline().decode().strip().split()
            for p in range(NP):
                Nodes[i][p]=linelist[p]
        self.SetNodes(Nodes,Types=self.PropTypes)
                
    def ReadElemsASCII(self,fp):
        '''Internal function to read the elements of a ply file in ASCII format'''
        Elems = np.zeros(self.NElem,dtype=[('p1','i4'),('p2','i4'),('p3','i4')])
        NP = 3
        for i in range(self.NElem):
            linelist=fp.readline().decode().strip().split()[1:]
            for p in range(NP):
                Elems[i][p]=linelist[p]
        self.SetElems(Elems)

    def ReadNodesBinary(self,fp):
        '''Internal function to read the nodes of a ply file in binary format'''
        Nodes = np.fromfile(fp,dtype=self.PropTypes.GetDtype(),count=self.NNodes)
        self.SetNodes(Nodes,Types=self.PropTypes)
                
    def ReadElemsBinary(self,fp):
        '''Internal function to read the elements of a ply file in binary format'''
        Elems = nprec.drop_fields(np.fromfile(fp,dtype=[('count', 'u1'),('p1','i4'),('p2','i4'),('p3','i4')],count=self.NElem),'count')
        self.SetElems(Elems)
         
        
    ## -- WRITE PLY OPERATIONS -- ##
    
    def SaveFile(self,plyfile,Endi=None,asASCII=False):
        '''Save mesh to ply file.
        
        plyfile - path to output plyfile
        Endi[None] - set the endienness of the binary output
        asASCII - output as ASCII text instead of binary (slower and larger).
        '''
        if Endi is not None:
            self.Endi = Endi
        if self.Endi == "=":
            self.Endi=SYS_ENDI
        self.NormEndi()
    
        with open(plyfile, 'wb') as fp:
            self.WriteHeader(fp,asASCII)
            if asASCII:
                self.WriteNodesASCII(fp)
                self.WriteElemsASCII(fp)
            else:
                self.WriteNodesBinary(fp)
                self.WriteElemsBinary(fp)
            
    def WriteHeader(self,fp,asASCII):
        '''Internal function to write the header of a ply file.
        
        fp - open file pointer
        asASCII - if True export data as ASCII instead of binary.
        '''
        propliststring = ""
        for plbl in self.PropTypes.Ltype:
            if not plbl == "N":
                propliststring += "property "+ numpy2ply(self.PropTypes.GetType(plbl)) +" "+ plbl +"\n"
        
        if asASCII:
            plyformat = "ascii"
        else:
            plyformat = GetEndiply(self.Endi)
        
        
        header= \
        "ply\n" \
        "format " + plyformat + " 1.0\n" \
        "element vertex "+str(self.NNodes)+"\n" \
        + propliststring + \
        "element face "+str(self.NElem)+"\n" \
        "property list uchar int vertex_indices\n" \
        "end_header\n"
        
        fp.write(header.encode())

    def WriteArray(self,fp,Array):
        '''Internal function to write an array in ASCII format'''
        for line in Array:
            asciiline = " ".join([str(el) for el in line]) + "\n"
            fp.write(asciiline.encode())
        
    def WriteNodesASCII(self,fp): 
        '''Internal function to write the Nodes in ASCII format'''
        NoNLabs=self.PropTypes.Ltype[1:]
        TmpNodes=self.Nodes.Mat[NoNLabs].copy()
        self.WriteArray(fp,TmpNodes)
        
    def WriteElemsASCII(self,fp):    
        '''Internal function to write the Elements in ASCII format'''
        tmpdtype = self.Elems.Mat.dtype.descr
        tmpdtype[0]=('N', self.Endi+'u1')
        TmpElems = self.Elems.Mat.astype(tmpdtype)
        TmpElems['N']=3
        self.WriteArray(fp,TmpElems)
        
    def WriteNodesBinary(self,fp):
        '''Internal function to write the Nodes in binary format'''
        NoNLabs = self.PropTypes.Ltype[1:]
        tmpdtype = self.Nodes.Mat.dtype.descr[1:]
        try:
            TmpNodes = self.Nodes.Mat[NoNLabs].copy().data.astype(tmpdtype)
        except:
            TmpNodes = self.Nodes.Mat[NoNLabs].copy().astype(tmpdtype)
        TmpNodes.tofile(fp)
        
    def WriteElemsBinary(self,fp):
        '''Internal function to write the Elements in binary format'''
        tmpdtype = self.Elems.Mat.dtype.descr
        tmpdtype[0]=('N', 'u1')
        TmpElems =self.Elems.Mat.astype(tmpdtype)
        TmpElems['N']=3
        TmpElems.tofile(fp)

# Class for VTU I/O        
class VTUIO(BaseIO):
    def __init__(self):
        '''Class to export data in VTU format (importing not implemented).

        The attribute self.VecLabs contains a list of the form ['Name',[labs]],
        where Name is a string that identifies a field and [labs] is a list that
        contains all the labels that makeup that field, in order.
        For example: ['displacement',['dx','dy','dz']]

        TODO: Maybe it would work better to ust the vtk module.
        '''

        BaseIO.__init__(self)
        self.VecLabs=[] # list of lists. Each item of VecLabs contains a list of the
                        # form ['Name',[labs]], where Name is a string that identifies
                        # a field and [labs] is a list that contains all the labels that
                        # makeup that field, in order.
                        #   For example: ['displacement',['dx','dy','dz']]

    def GetNodesString(self,Labs):
        string=str()
        for line in self.Nodes.Mat[Labs].copy():
            string += " ".join([str(el) for el in line]) + "\n"
        return string
    
    def GetElemsStrings(self):
        conn=str()
        off=str()
        typs=str()
        n=0
        NSIDES=3  #Triangle
        VTKTYPE=5 #Triangle
        for line in self.Elems.Mat[["p1","p2","p3"]].copy():
            n+=NSIDES
            conn += " ".join([str(el) for el in line]) + "\n"
            off  += str(n) + " "
            typs += str(VTKTYPE) + " "
        return conn,off,typs
    
    ## -- WRITE VTU OPERATIONS -- ##
    
    def SaveFile(self,vtufile,VecLabs=[],Endi=None):
        '''Save mesh to vtu file.
        
        vtufile - path to output vtufile
        VecLabs[[]] - list of grouped nodal that form higher dimension fields (see VTUIO doc for more)
        Endi[None] - set the endienness of the binary output
        '''
        if Endi is not None:
            self.Endi = Endi
        if self.Endi == "=":
            self.Endi=SYS_ENDI
        self.NormEndi()
    
        self.SetPiece()
        self.SetNodes()
        self.SetElems()
        
        if self.VecLabs==[]:
            print("VecLabs is empty. Only geometry exported. Do self.VecLabs.append(['Name',[labs]]) to add data")
        
        self.SetNodeData()
        self.SetElemData()
        
        with open(vtufile, 'w') as fp:
            self.doc.writexml(fp, newl='\n')
    
    def SetPiece(self):
        '''Internal function to generate vtu data structure'''
        # Root element
        self.doc = xml.dom.minidom.Document()
        root_element = self.doc.createElementNS("VTK", "VTKFile")
        root_element.setAttribute("type", "UnstructuredGrid")
        root_element.setAttribute("version", "0.1")
        root_element.setAttribute("byte_order", "LittleEndian")
        self.doc.appendChild(root_element)
        # Unstructured grid element
        unstructuredGrid = self.doc.createElementNS("VTK", "UnstructuredGrid")
        root_element.appendChild(unstructuredGrid)
        # Piece 0 (only one)
        self.piece = self.doc.createElementNS("VTK", "Piece")
        self.piece.setAttribute("NumberOfPoints", str(self.NNodes))
        self.piece.setAttribute("NumberOfCells", str(self.NElem))
        unstructuredGrid.appendChild(self.piece)    
    
    def SetNodes(self):
        '''Internal function to generate vtu data structure'''
        ### Points ####
        points = self.doc.createElementNS("VTK", "Points")
        self.piece.appendChild(points)

        # Point location data
        point_coords = self.doc.createElementNS("VTK", "DataArray")
        point_coords.setAttribute("type", "Float32")
        point_coords.setAttribute("format", "ascii")
        point_coords.setAttribute("NumberOfComponents", "3")
        points.appendChild(point_coords)
        
        string = self.GetNodesString(['x', 'y', 'z'])
        point_coords_data = self.doc.createTextNode(string)
        point_coords.appendChild(point_coords_data)
    
    def SetElems(self):
        '''Internal function to generate vtu data structure'''
        #### Cells ####
        cells =self.doc.createElementNS("VTK", "Cells")
        self.piece.appendChild(cells)

        # Cell locations
        conn,off,typs = self.GetElemsStrings() #Location Data
        
        cell_connectivity = self.doc.createElementNS("VTK", "DataArray")
        cell_connectivity.setAttribute("type", "Int32")
        cell_connectivity.setAttribute("Name", "connectivity")
        cell_connectivity.setAttribute("format", "ascii")        
        cells.appendChild(cell_connectivity)
        connectivity = self.doc.createTextNode(conn)
        cell_connectivity.appendChild(connectivity)

        cell_offsets = self.doc.createElementNS("VTK", "DataArray")
        cell_offsets.setAttribute("type", "Int32")
        cell_offsets.setAttribute("Name", "offsets")
        cell_offsets.setAttribute("format", "ascii")                
        cells.appendChild(cell_offsets)
        offsets = self.doc.createTextNode(off)
        cell_offsets.appendChild(offsets)

        cell_types = self.doc.createElementNS("VTK", "DataArray")
        cell_types.setAttribute("type", "UInt8")
        cell_types.setAttribute("Name", "types")
        cell_types.setAttribute("format", "ascii")                
        cells.appendChild(cell_types)
        types = self.doc.createTextNode(typs)
        cell_types.appendChild(types)

    def SetNodeData(self):
        '''Internal function to generate vtu data structure'''
        point_data = self.doc.createElementNS("VTK", "PointData")
        self.piece.appendChild(point_data)
        
        for Lab in self.VecLabs:
            VName=Lab[0]
            labs=Lab[1]
            
            forces = self.doc.createElementNS("VTK", "DataArray")
            forces.setAttribute("Name", VName)
            forces.setAttribute("NumberOfComponents", str(len(labs)))
            forces.setAttribute("type", "Float32")
            forces.setAttribute("format", "ascii")
            point_data.appendChild(forces)

            string = self.GetNodesString(labs)          
            forceData = self.doc.createTextNode(string)
            forces.appendChild(forceData)

    def SetElemData(self):
        '''Internal function to generate vtu data structure'''
        cell_data = self.doc.createElementNS("VTK", "CellData")
        self.piece.appendChild(cell_data)
        

# Class for FEBio I/O
class FEBioIO(BaseIO):
    def __init__(self):
        '''Class to export data in FEBio format (importing not implemented).

        Special loading conditions and material properties are defined when
        calling the SaveFile function.
        '''
        BaseIO.__init__(self)
    
    ## -- WRITE FEBIO OPERATIONS -- ##

    # Aux operations
    def createChild(self,parent,child):
        childobjg = self.doc.createElement(child)
        parent.appendChild(childobjg)
        return childobjg

    def addText(self,parent,text):
        textobj = self.doc.createTextNode(text)
        parent.appendChild(textobj)
        return textobj

    def addObjectsAsChildren(self,pobject,pdict):
        for key in pdict:
            keyobj = self.doc.createElement(key)
            valtxtobj = self.doc.createTextNode(pdict[key])
            keyobj.appendChild(valtxtobj)
            pobject.appendChild(keyobj)

    # Aux Functions

    def processdata(self,fibertype=0,**kwargs):
        self.MeshClass.ComputeNormals()
        if fibertype==2:
            self.MeshClass.ComputeCurvatures()
            self.MeshClass.SmoothCurvatures(5)
            self.MeshClass.ComputeCurvaturePrincipalDirections()
            Director_Vector_List = [
                [[551,1204,707],[0.55,-0.17,0.82]],
                [[767,1015,759],[0.07,0.75,0.65]],
                [[995,1094,740],[-0.43,0.57,0.7]],
                [[407,943,402],[0.78,-0.53,0.32]],
                [[778,689,463],[0.41,0.66,0.63]],
                [[1066,709,442],[-0.02,0.85,0.52]],
                [[373,442,154],[0.8,-0.6,0.08]],
                [[708,239,206],[0.64,0.66,0.38]],
                [[965,313,263],[0.27,0.9,0.35]],
            ]
            Aux_Vec = [-0.15,0.88,0.45]            
            self.MeshClass.ReadjustMinCurvDirections(Director_Vector_List=Director_Vector_List)
            self.MeshClass.FlipMinCurvDirections(Aux_Director_Vector=Aux_Vec)

        self.makefiberangles(fibertype=fibertype,**kwargs)

    def makefiberangles(self,fibertype=0,linfuncparam=(-0.0835,156.738)):
        '''Compute angle of fibers for each element.
        fibertype default = 0
            0- aligned with X (like 3 with a,b=0)
            1- random
            2- direction of zero curvature
            3- linear function (linfuncparam = [a,b] angle = a*y_coord+b)
            4- 90 degress from 3
        '''
        Nel = len(self.Elems.Mat)
        #self.Angles = np.array((len(self.Elems.Mat)),dtype='f8')
        if fibertype==0: # Aligned with X,Y
            self.Angles = np.zeros((Nel),dtype='f8')
        elif fibertype==1: # Random
            self.Angles = np.random.random((Nel))*180
        elif fibertype==2: # Dir Zero Curv
            self.Angles = np.zeros((Nel),dtype='f8')
            Vmm=np.array([0.0,0.0,0.0])
            for el in range(Nel):
                Vmm*=0.0
                for ni in self.MeshClass.EN[el]:
                    Vmm+=self.MeshClass.NMinMag[ni]
                angle = np.degrees(np.arctan2(Vmm[1],Vmm[0]))
                if angle<0: angle+=360
                if angle>180: angle-=180
                self.Angles[el]=angle
        elif fibertype==3 or fibertype==4 : # Linear function
            self.Angles = np.zeros((Nel),dtype='f8')
            a,b = linfuncparam
            for el in range(Nel):
                y_coord = 0.0
                for ni in self.MeshClass.EN[el]:
                    y_coord+=self.Nodes.Mat[ni]["y"]
                y_coord/=3.0
                angle = y_coord*a+b
                if fibertype==4:angle-=90.0
                self.Angles[el]=angle

    # Add Parts to feb File

    def addBeginning(self):
        # Module
        Module = self.createChild(self.febio_spec,"Module")
        Module.setAttribute("type", "solid")

        # Globals
        Globals = self.createChild(self.febio_spec,"Globals")
        Constants = self.createChild(Globals,"Constants")
        self.addObjectsAsChildren(Constants,{"T":"0","R":"0","Fc":"0"})

    def addMaterial(self,prestrain=False,dispersion=False,kip=0.15833):
        # Material
        Material = self.createChild(self.febio_spec,"Material")

        # material 1
        material1 = self.createChild(Material,"material")
        material1.setAttribute("id", "1")
        material1.setAttribute("name", "Material1")

        if prestrain:
            material1.setAttribute("type", "uncoupled prestrain elastic")
            prestrain = self.createChild(material1,"prestrain")
            prestrain.setAttribute("type", "prestrain gradient")
            elastic = self.createChild(material1,"elastic")

        else:
            elastic = material1

        elastic.setAttribute("type", "uncoupled solid mixture")
        # Matrix
        solidnh = self.createChild(elastic,"solid")
        solidnh.setAttribute("type", "Mooney-Rivlin")
        self.addObjectsAsChildren(solidnh,{"c1":"0.08","c2":"0.0","k":"1000"}) #E=2MPa => G=0.66uN/um2 => c1=0.33

        # Fibers
        solidf = self.createChild(elastic,"solid")
        if dispersion:
            solidf.setAttribute("type", "hdispfibers")
            self.addObjectsAsChildren(solidf,{"k1":"0.08","k2":"200","kip":str(kip),"kop":"0.5"})
        else:
            solidf.setAttribute("type", "fiber-exp-pow-uncoupled")
            self.addObjectsAsChildren(solidf,{"ksi":"0.08","alpha":"200","beta":"2.0","theta":"0.0","phi":"90.0"})

    def addGeometry(self):
        # Geometry
        Geometry = self.createChild(self.febio_spec,"Geometry")

        #Nodes
        Nodes = self.createChild(Geometry,"Nodes")
        Nodes.setAttribute("name","Object01")
        for nd in range(len(self.Nodes.Mat)):
            node_coords = self.createChild(Nodes,"node")
            node_coords.setAttribute("id",str(nd+1))
            string=",".join([str(self.Nodes.Mat[nd][lab]) for lab in ["x","y","z"]])
            self.addText(node_coords,string)

        #Elements
        Elements = self.createChild(Geometry,"Elements")
        Elements.setAttribute("type","tri3")
        Elements.setAttribute("name","Part1")
        Elements.setAttribute("mat","1")
        for el in range(len(self.Elems.Mat)):
            elem_nodes = self.createChild(Elements,"elem")
            elem_nodes.setAttribute("id",str(el+1))
            string=",".join([str(self.Elems.Mat[el][lab]+1) for lab in ["p1","p2","p3"]])
            self.addText(elem_nodes,string)

        #NodeSet 
        NodeSet = self.createChild(Geometry,"NodeSet")
        NodeSet.setAttribute("name","FixedDisplacement1")
        for nd in range(len(self.Nodes.Mat)):
            if self.MeshClass.NIsB[nd]:
                node = self.createChild(NodeSet,"node")
                node.setAttribute("id",str(nd+1))

        #Surface
        Surface = self.createChild(Geometry,"Surface")
        Surface.setAttribute("name","PressureLoad1")
        for el in range(len(self.Elems.Mat)):
            elem_nodes = self.createChild(Surface,"tri3")
            elem_nodes.setAttribute("id",str(el+1))
            string=",".join([str(self.Elems.Mat[el][lab]+1) for lab in ["p1","p2","p3"]])
            self.addText(elem_nodes,string)

    def addMeshData(self,thickness=30.0,prestrain=False):
        '''thickness default = 30 um
        '''
        
        # MeshData
        MeshData = self.createChild(self.febio_spec,"MeshData")

        #ElementData - shell thickness
        ElementData = self.createChild(MeshData,"ElementData")
        ElementData.setAttribute("var","shell thickness")
        ElementData.setAttribute("elem_set","Part1")
        for el in range(len(self.Elems.Mat)):
            elem_thick = self.createChild(ElementData,"elem")
            elem_thick.setAttribute("lid",str(el+1))
            self.addText(elem_thick,str(thickness)+","+str(thickness)+","+str(thickness))


        #ElementData - mat_axis (Fiber Orientation)
        ElementData = self.createChild(MeshData,"ElementData")
        ElementData.setAttribute("var","mat_axis")
        ElementData.setAttribute("elem_set","Part1")

        for el in range(len(self.Elems.Mat)):
            elem_mat_axis = self.createChild(ElementData,"elem")
            elem_mat_axis.setAttribute("lid",str(el+1))

            elem_mat_axis_a = self.createChild(elem_mat_axis,"a")
            elem_mat_axis_d = self.createChild(elem_mat_axis,"d")

            angle = self.Angles[el]
            normal = self.MeshClass.ENorms[el]
            normal/=np.linalg.norm(normal)
            V=np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0])
            nz = np.array([0,0,1])
            avec = V-np.dot(V,normal)/np.dot(nz,normal)*nz
            avec/=np.linalg.norm(avec)
            dvec=np.cross(normal,avec)

            self.addText(elem_mat_axis_a,",".join([str(x) for x in avec]))
            self.addText(elem_mat_axis_d,",".join([str(x) for x in dvec]))

        if prestrain:
            #ElementData - Prestrain Deformation
            ElementData = self.createChild(MeshData,"ElementData")
            ElementData.setAttribute("var","F0")
            ElementData.setAttribute("elem_set","Part1")

            for el in range(len(self.Elems.Mat)):
                elem = self.createChild(ElementData,"elem")
                elem.setAttribute("lid",str(el+1))
                #elem_F0 = self.createChild(elem,"F0")

                angle = self.Angles[el]
                normal = self.MeshClass.ENorms[el]
                normal/=np.linalg.norm(normal)
                V=np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0])
                nz = np.array([0,0,1])
                avec = V-np.dot(V,normal)/np.dot(nz,normal)*nz
                avec/=np.linalg.norm(avec)
                dvec=np.cross(normal,avec)

                Q=np.stack((avec,dvec,normal),axis=1)
                lam=1.0
                Fs=np.array([
                    [lam,0,0],
                    [0,lam,0],
                    [0,0,1/(lam**2)]
                ])
                F=Q@Fs@Q.T

                text=",".join([  ",".join([str(F[i,j]) for j in range(3)])  for i in range(3)])
                text="1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0"
                print("WARNING: OVERIDDEN F TO IDENTITY FOR TESTING")
                self.addText(elem,text)

    def addBoundary(self):
        # Boundary
        Boundary = self.createChild(self.febio_spec,"Boundary")
        fixedbc = self.createChild(Boundary,"fix")
        fixedbc.setAttribute("bc","x,y,z")
        fixedbc.setAttribute("node_set","FixedDisplacement1")        

    def addConstraints(self,prestrain=False):
        # Constraints
        if prestrain:
            Constraints = self.createChild(self.febio_spec,"Constraints")
            constraint = self.createChild(Constraints,"constraint")
            constraint.setAttribute("type","prestrain")
            self.addObjectsAsChildren(constraint,{"tolerance":"0.1"})
            self.addObjectsAsChildren(constraint,{"min_iters":"3"})

    def addLoadData(self):
        # LoadData
        LoadData = self.createChild(self.febio_spec,"LoadData")
        
        loadcurve1 = self.createChild(LoadData,"loadcurve")
        loadcurve1.setAttribute("id","1")
        loadcurve1.setAttribute("type","smooth")
        self.addObjectsAsChildren(loadcurve1,{"point":"0,0"})
        self.addObjectsAsChildren(loadcurve1,{"point":"1,1"})

        loadcurve2 = self.createChild(LoadData,"loadcurve")
        loadcurve2.setAttribute("id","2")
        loadcurve2.setAttribute("type","step")
        for i in range(11):
            f=i/10
            if i==0:
                self.addObjectsAsChildren(loadcurve2,{"point":"0,0"})
            else:
                self.addObjectsAsChildren(loadcurve2,{"point":str(f)+",0.1"})

    def addOutput(self):
        # Output
        Output = self.createChild(self.febio_spec,"Output")
        plotfile = self.createChild(Output,"plotfile")
        plotfile.setAttribute("type","febio")
        
        for outdata in ["displacement","stress","Lagrange strain"]:
            var = self.createChild(plotfile,"var")
            var.setAttribute("type",outdata)

    def addStep(self,load=2E-3): 
        'Default load = 2E-3 = 2MPa'
        # Step1
        Step1 = self.createChild(self.febio_spec,"Step")
        Step1.setAttribute("name","Step1")

        #Control
        Control = self.createChild(Step1,"Control")

        StepParam = {
            "time_steps":"100",
            "step_size":"0.01",
            "max_refs":"15",
            "max_ups":"10",
            "diverge_reform":"1",
            "reform_each_time_step":"1",
            "dtol":"0.001",
            "etol":"0.01",
            "rtol":"0",
            "lstol":"0.9",
            "min_residual":"1e-020",
            #"plot_level":"PLOT_MUST_POINT",
            "qnmethod":"0"}
        self.addObjectsAsChildren(Control,StepParam)

        time_stepper = self.createChild(Control,"time_stepper")
        self.addObjectsAsChildren(time_stepper,{"dtmin":"0.00001"})
        dtmax = self.createChild(time_stepper,"dtmax")
        dtmax.setAttribute("lc","2")
        self.addText(dtmax,"0.2")
        TimeParam = {
            "max_retries":"10",
            "opt_iter":"10",
            "aggressiveness":"1"}
        self.addObjectsAsChildren(time_stepper,TimeParam)

        analysis = self.createChild(Control,"analysis")
        analysis.setAttribute("type","static")


        #Loads
        Loads = self.createChild(Step1,"Loads")

        surface_load = self.createChild(Loads,"surface_load")
        surface_load.setAttribute("type","pressure")
        surface_load.setAttribute("surface","PressureLoad1")

        pressure = self.createChild(surface_load,"pressure")
        pressure.setAttribute("lc","1")
        self.addText(pressure,str(load))

        linear = self.createChild(surface_load,"linear")
        self.addText(linear,"0")
      
    def SaveFile(self,febiofile,Endi=None,fibertype=0,prestrain=False,dispersion=False,kip=0.15833,load=-2E-2):
        '''Save mesh to FEBio file.
        
        febiofile - path to output febiofile
        Endi[None] - set the endienness of the binary output
        fibertype[0] - type of fiber alignement. (see makefiberangles() doc for more)
        prestrain[False] - include prestrain (requires Prestrain plugin)
        dispersion[False] - include dispersion (required HDisp plugin)
        kip[0.15833] - in-plane dispersion parameter (when dispersion=True)
        load[-2E-2] - value of load pressure in MPa.
        '''
        if Endi is not None:
            self.Endi = Endi
        if self.Endi == "=":
            self.Endi=SYS_ENDI
        self.NormEndi()
    
        print("Process necessary geometrical data ...")
        self.processdata(fibertype=fibertype)

        print("Begin creating FEBio File ...")
        self.doc = xml.dom.minidom.Document()
        self.febio_spec = self.createChild(self.doc,"febio_spec")
        self.febio_spec.setAttribute("version", "2.5")

        self.addBeginning()
        self.addMaterial(prestrain=prestrain,dispersion=dispersion,kip=kip)
        
        print("Create Geometry ...")
        self.addGeometry()
        print("Create MeshData ...")
        self.addMeshData(prestrain=prestrain)
        print("Finish BCs and loadings ...")
        self.addBoundary()
        self.addConstraints(prestrain=prestrain)
        self.addLoadData()
        self.addOutput()
        self.addStep(load=load)

        with open(febiofile, 'w') as fp:
            self.doc.writexml(fp, addindent="    ", newl='\n')
    






# AUX FUNCTIONS FOR MY MESH
@nb.jit(nopython=True)
def _Aux_getBaricentricCoordinates(point,P1,P2,P3):
        def mycross(r1,r2):
            return r1[0]*r2[1]-r1[1]*r2[0]
        r12 = P2-P1
        r23 = P3-P2
        r3p = point-P3
        r2p = point-P2
        iA = 1/mycross(r12,r23)    
        l1 = mycross(r23,r3p)*iA   
        l3 = mycross(r12,r2p)*iA
        return l1,1-l1-l3,l3


#@nb.jit(nopython=True)
@nb.jit('f8[:,:](f8[:,:],f8,int32,f8[:,:],int32[:,:])')
def _Aux_RenderTriangles(picture,scale,NElems,NodesMat,EN):
    for nel in range(NElems):
        P1=NodesMat[EN[nel,0],:]
        P2=NodesMat[EN[nel,1],:]
        P3=NodesMat[EN[nel,2],:]

        Pmin = np.floor(np.amin([P1,P2,P3],axis=0)/scale).astype(int)
        Pmax = np.ceil(np.amax([P1,P2,P3],axis=0)/scale+1).astype(int)

        for xp in range(Pmin[0],Pmax[0]):
            for yp in range(Pmin[1],Pmax[1]):
                l1,l2,l3 = _Aux_getBaricentricCoordinates(np.array([xp*scale,yp*scale]),P1[0:2],P2[0:2],P3[0:2])
                if (l1>=0 and l2>=0 and l3>=0 and l1<=1 and l2<=1 and l3<=1): #Is inside Triangle
                    for node,bari in zip([P1,P2,P3],[l1,l2,l3]):
                        picture[xp,yp]+=node[2]*bari
 
    return picture      

class MyMesh(object):
    def __init__(self,Nodes,Elems):
        '''
        Class for mesh manipulation. Contains algorithms to compute Normals and Curvatures.

        Nodes - CNodes object
        Elements - CElements object
        '''
        self.NNodes = Nodes.GetNNodes() # Number of Nodes
        self.NElems = Elems.GetNElems() # Number of Elements
        self.Nodes  = Nodes # CNodes object
        self.Elems  = Elems # CElements object
        self.IndxL = self.MakeIndexed() # List that relates node number with node index (useful when node numbers are not their indexes in self.Nodes)
        
        # Neighbours in 1-Ring
        self.NN = np.empty(self.NNodes,dtype=object) # Nodes near Node (list of nodes that are 1 side away from node)
        self.NE = np.empty(self.NNodes,dtype=object) # Elements near Node (list of elements touching the node)
        self.EN = np.empty(self.NElems,dtype=object) # Nodes near Element (list of nodes touching the element)
        self.EE = np.empty(self.NElems,dtype=object) # Elements near Element (list of elements touching the nodes of the element)
        self.ComputeNeighbours()
        
        # Sides in 1-Ring
        #self.Sides=np.zeros(self.NSides,dtype=[('ni','i4'),('nf','i4'),('ei','i4'),('ef','i4'),('beta','f8')])
        self.NS = np.empty(self.NNodes,dtype=object) # Sides per Node
        self.ES = np.zeros((self.NElems,3),dtype='i4') # Sides per Element
        self.ComputeSides()
        
        # Check if nodes, elements or sides are in boundary of mesh
        self.NIsB = np.zeros(self.NNodes,dtype=np.bool) # Is Boundary per Node
        self.EIsB = np.zeros(self.NElems,dtype=np.bool) # Is Boundary per Element
        self.SIsB = np.zeros(self.NSides,dtype=np.bool) # Is Boundary per Side
        self.ComputeBoundary()

    def MakeIndexed(self):
        '''Internal function to sort the node numbers'''
        self.NewElemsMat=np.zeros((self.NElems,3),dtype='i4')
        # "Allocate empty structured array"
        IndxL=np.empty(self.NNodes,dtype=[('N','i4'),('I','i4')])
        
        # "Fill N and I"
        IndxL['N'] = self.Nodes.Mat['N'] # Original node numbering
        IndxL['I'] = np.arange(self.NNodes) # Sorted sequential index
        
        # Sort
        IndxL.sort(order='N')
        
        # Add Ip1, Ip2 and Ip3
        idxp1 = IndxL['N'].searchsorted(self.Elems.Mat['p1']) # Find indexes of p1 of all elements
        idxp2 = IndxL['N'].searchsorted(self.Elems.Mat['p2']) # Find indexes of p2 of all elements
        idxp3 = IndxL['N'].searchsorted(self.Elems.Mat['p3']) # Find indexes of p3 of all elements
        self.NewElemsMat[:,0] = IndxL['I'][idxp1] # Setup First node in Index numbering
        self.NewElemsMat[:,1] = IndxL['I'][idxp2] # Setup Second node in Index numbering
        self.NewElemsMat[:,2] = IndxL['I'][idxp3] # Setup Third node in Index numbering

        return IndxL
    
    def ComputeNeighbours(self):
        '''
        Internal function to compute all the arrays with neighbours in 1-Ring.
            self.NN - Nodes near Node (list of nodes that are 1 side away from node)
            self.NE - Elements near Node (list of elements touching the node)
            self.EN - Nodes near Element (list of nodes touching the element)
            self.EE - Elements near Element (list of elements touching the nodes of the element)
        '''

        # Reset
        for i in range(self.NNodes):
            self.NE[i]=[]
            self.NN[i]=[]
        for e in range(self.NElems):
            self.EN[e]=[]
            self.EE[e]=[]
            
        # Compute EN and NE
        for e in range(self.NElems):
            self.EN[e]=self.NewElemsMat[e]
            for p in self.EN[e]:
                self.NE[p].append(e)
        
        # Compute NN
        for i in range(self.NNodes): #For all nodes
            for ei in self.NE[i]: #For all elements touching that node
                for nei in self.EN[ei]: #For all nodes of that element
                    if (nei not in self.NN[i]) and (nei != i): self.NN[i].append(nei)

        # Compute EE    
        for e in range(self.NElems): #For all elements    
            for ne in self.EN[e]: #For all nodes of that element
                for ene in self.NE[ne]: #For all elements touching that node
                    if ene not in self.EE[e]: self.EE[e].append(ene)

    def ComputeSides(self):
        '''Internal function to compute sides of triangles of the mesh.

        self.Sides - dtype:
            'ni' - first node
            'nf' - second node
            'ei' - first element
            'ef' - second element
            'beta' - dihedral angles
            'nEE' - side length
            'S' - shape operator

        self.NS - Sides per Node
        self.ES - Sides per Element
        '''

        # Reset
        for i in range(self.NNodes):
            self.NS[i]=[]

        # Compute NS and Sides(ni,nf)
        GSides=0
        STemp=np.zeros((self.NNodes*10,2))
        for n in range(self.NNodes):
            for nd in self.NN[n]:
                if n<nd:
                    STemp[GSides,0]=n
                    STemp[GSides,1]=nd
                    self.NS[n].append(GSides)
                    self.NS[nd].append(GSides)
                    GSides+=1
        self.NSides=GSides
        self.Sides=np.zeros(self.NSides,dtype=[('ni','i4'),('nf','i4'),('ei','i4'),('ef','i4'),('beta','f8'),('nEE','f8'),('S',('f8',(3,3)))])
        self.Sides['ni']=STemp[:self.NSides,0]
        self.Sides['nf']=STemp[:self.NSides,1]  

        
        # Compute ES and Sides(ei,ef)                 
        self.Sides['ei']=-1
        self.Sides['ef']=-1
        for e in range(self.NElems): #For all elements
            p1,p2,p3=self.EN[e]
            S1=self.FindSide(p1,p2)
            S2=self.FindSide(p1,p3)
            S3=self.FindSide(p2,p3)
            self.ES[e,:]=[S1,S2,S3]
            for s in self.ES[e,:]:
                if self.Sides[s]['ei']==-1:
                    self.Sides[s]['ei']=e
                elif self.Sides[s]['ef']==-1:
                    self.Sides[s]['ef']=e
                else:
                    raise ValueError("Mesh in not 2-Manifold")
                    
    def FindSide(self,n1,n2):
        '''Find side number that connects n1 and n2'''
        if n1==n2: raise ValueError("Cannot find side of same node")
        for val in self.NS[n1]:
            if val in self.NS[n2]:
                return val
        raise ValueError("Cannot find side for node pair")
            
    def ComputeBoundary(self):
        '''Compute boundary arrays.

        self.NIsB - Is Boundary per Node
        self.EIsB - Is Boundary per Element
        self.SIsB - Is Boundary per Side
        '''

        for si in range(self.NSides):
            ni,nf,ei,ef = [self.Sides[si][lab] for lab in ['ni','nf','ei','ef']]
            if ef==-1: # If the side only has one triangle, then it is at a boundary
                self.NIsB[ni]=True
                self.NIsB[nf]=True
                self.EIsB[ei]=True
                self.SIsB[si]=True
                
    def ComputeNormals(self,mode=0):
        ''' Compute normals of the mesh. Normals are computed from 
        averaging element normals. Averaging can be weighted by the
        area or the angle of the element that originated it.

        Mode - Type of Nodal normal computation 
            0-Simple Average
            1-Weighted by Area
            2-Weighted by Angle
        '''

        # local function that determines the weight based on mode
        def mymode(e,nei):
            if mode==0:
                return 1.0
            elif mode==1:
                return self.EAreas[e]
            elif mode==2:
                return self.EAngs[e,nei]

        self.ENorms = np.empty((self.NElems,3),dtype='f8') # Element Normals - nx,ny,nz
        self.EAngs  = np.empty((self.NElems,3),dtype='f8') # Element Angles - a1,a2,a3
        self.EAreas = np.empty(self.NElems,dtype='f8') # Element Areas - A
        self.NNorms = np.zeros((self.NNodes,3),dtype='f8') # Node Normals - nx,ny,nz
        
        for e in range(self.NElems):
            #Triangle Points and Vectors
            P1,P2,P3 = [np.array([P['x'],P['y'],P['z']]) for P in self.Nodes.Mat[self.EN[e]]]
            V=P2-P1
            W=P3-P1
            Z=P3-P2
            
            #Element Normal
            self.ENorms[e,:] = np.cross(V,W)
            #Element Area
            self.EAreas[e]=np.linalg.norm(self.ENorms[e,:])/2
            #Normalize Element Normal
            self.ENorms[e,:]/=(2*self.EAreas[e])
            
            #Internal Angles
            nV=np.linalg.norm(V)
            nW=np.linalg.norm(W)
            nZ=np.linalg.norm(Z)
            x=np.dot(V,W)/(nV*nW)
            self.EAngs[e,:]=np.arccos([x,(-x*nW+nV)/nZ,(-x*nV+nW)/nZ])

            #Assign Node Normal and Node Areas
            for nei,n in enumerate(self.EN[e]):
                self.NNorms[n,:]+=self.ENorms[e,:]*mymode(e,nei)
                
        #Final Nodal Operations
        for n in range(self.NNodes):
            self.NNorms[n,:]/=np.linalg.norm(self.NNorms[n,:]) # Normalize Node Normals

    def GetCentroid(self,e):
        '''Get centroid of element'''
        C=np.zeros(3)
        for P in self.Nodes.Mat[self.EN[e]]:
            C+=[P['x'],P['y'],P['z']]
        return C/3.0    

    def InterceptVerticalSlice(self,P1,P2):
        '''Get line resulting from slicing the surface with a vertical plane that passes through P1 and P2

        P1 - vector/list/tuple with x and y coordinates of first point
        P2 - vector/list/tuple with x and y coordinates of second point
        '''

        # Get Plane Equation
        a,b = P1[1]-P2[1],P2[0]-P1[0]
        d=a*P1[0]+b*P1[1]
        def plane(p):
            '''Distance of point p to plane. Sign reflects
            which side of the plane the point is.'''
            return a*p[0]+b*p[1]-d
        def compare(pi,pf):
            ''' Find position of plane with respect to the line that
            connects pi to pf. Returns tha value of alpha, where alpha
            is the distance between the intersecting point and pi, normalized
            by the distance between pi and pf. Alpha is 0 if the plane is at pi,
            1 if the plane is at pf, [0,1] if the plane is between these two
            points and None if the plane is not between the two points.'''
            pli=plane(pi)
            if pli==0.0: return 0.0 # Plane cuts at pi
            plf=plane(pf)
            if plf==0.0: return 1.0 # Plane cuts at pf
            if pli*plf > 0.0: return None  # Plane cuts outside pi-pf
            return pli/(pli-plf)

        # Find Starting edge
        SideStart = None
        for si in range(self.NSides):
            if not self.SIsB[si]:
                continue
            ni,nf,ei,ef = [self.Sides[si][lab] for lab in ['ni','nf','ei','ef']] # Get nodes and elements of the side
            #pi,pf = [(P['x'],P['y']) for P in [self.Nodes.Mat[nx] for nx in [ni,nf]]]
            pi,pf = self.Nodes.GetNumpyPoints([ni,nf])
            alpha = compare(pi,pf)
            if alpha is not None: # Plane does not cut outside pi-pf
                SideStart=si
                break
        
        # Raise Error if did not find starting edge
        if SideStart is None:
            raise ValueError("Plane does not intercept Edge of Membrane")
        
        # Get interpolated point
        def GetInterpPoint(side):
            '''Function to get interpolated point of a side.
            Returns None if plane does not cut that side and 
            returns the interception point otherwise.'''
            ni,nf = [self.Sides[side][lab] for lab in ['ni','nf']] # Get nodes of the side
            pi,pf = self.Nodes.GetNumpyPoints([ni,nf]) # Get node coordinates
            alpha = compare(pi,pf) # compute plane position
            if alpha is None:
                return None
            else:
                return pi*(1-alpha) + pf*alpha


        # March along elements that are cut by the slice
        cside = SideStart # current side
        pelem = -1 # previous element (first "previous" element of the side is the one outside the mesh [-1])
        SlicePoints = [GetInterpPoint(cside)] # list of point on the slice
        notOnEdge = True # Will march along the mesh until it finds an edge
        while notOnEdge:
            ei,ef = [self.Sides[cside][lab] for lab in ['ei','ef']] # Get elements of the side
            celem = ef if ei==pelem else ei # current element (the one connecting to the current side that is not the previous element)
            for sd in self.ES[celem,:]: # loop through all the element sides
                if sd==cside: # skip current side
                    continue
                P=GetInterpPoint(sd) # get interception of the side with the plane
                if P is not None: # If the plane is cutting the side
                    SlicePoints.append(P)
                    cside=sd
                    pelem=celem
                    break
            
            if self.SIsB[cside]: # check if current side (the new one) is on the edge
                notOnEdge=False

        return SlicePoints

    def MakePicture(self,resolution=(512,512),simulate=False):
        '''Render an image based on the mesh.

        resolution[(512,512)] - tuple with the dimensions of the image
        simulate[False] - set to true to skip rendering for faster testing
        '''
        picture=np.zeros(resolution)

        maxx=np.amax(self.Nodes.Mat['x'])
        maxy=np.amax(self.Nodes.Mat['y'])
        resx,resy = resolution
        scale=max(maxx/resx,maxy/resy) #micron/pixel
        
        # Make arrays in basic numpy format to speed up the rendering with numba
        Nodes_temp = np.array([self.Nodes.Mat['x'],self.Nodes.Mat['y'],self.Nodes.Mat['z']]).T
        EN_temp=np.reshape(np.concatenate(np.array(self.EN)),(-1,3))
        if not simulate: picture = _Aux_RenderTriangles(picture,scale,self.NElems,Nodes_temp,EN_temp)
        imgpix=MAIP.np2Image(picture)

        return imgpix,scale

    def ComputeCurvatures(self):
        '''Function to compute curvatures of the mesh. Requires computing the normals beforehand.
        Results are stored in internal vectors as follows:
            self.NHv - Mean Curvature Normal Vector
            self.NH - Mean Curvature
            self.NK -  Gaussian Curvature
            self.NSdihe - Structure Tensor (from dihedral)
            self.NSdiheS - Smoothed Structure Tensor (from dihedral) [smoothing later]
            self.NHdihe -  Mean Curvature (from dihedral)
        '''


        # Check if Normals were ran
        if not hasattr(self, 'EAngs'):
            raise RuntimeError("Please run ComputeNormals() before running ComputeCurvatures()")
            

        self.NAmix  = np.zeros(self.NNodes,dtype='f8') #A
        self.NARing = np.zeros(self.NNodes,dtype='f8') #A
        self.NAreaG = np.empty(self.NNodes,dtype='f8') #A
        self.NHv    = np.zeros((self.NNodes,3),dtype='f8') #Hnx,Hny,Hnz
        self.NH     = np.zeros(self.NNodes,dtype='f8') #H
        self.NK     = np.zeros(self.NNodes,dtype='f8') #K
        self.NSdihe = np.zeros(self.NNodes,dtype=('f8',(3,3))) #S (3x3)
        self.NSdiheS= np.zeros(self.NNodes,dtype=('f8',(3,3))) #S Smoothed (3x3)
        self.NHdihe = np.zeros(self.NNodes,dtype='f8') #Hdihe

        
        #Initialize NAreaG
        self.NAreaG[:] = 2*np.pi
        
        for e in range(self.NElems):
            #Triangle Points and Vectors
            P1,P2,P3 = [np.array([P['x'],P['y'],P['z']]) for P in self.Nodes.Mat[self.EN[e]]]
            V=P2-P1
            W=P3-P1
            Z=P3-P2
            
            nV=np.linalg.norm(V)
            nW=np.linalg.norm(W)
            nZ=np.linalg.norm(Z)

            #Mixed Area Factors
            IsObtuse=self.EAngs[e,:]>np.pi/2
            tangs=np.tan(self.EAngs[e,:])
            TV,TW,TZ = [1/tangs[2],1/tangs[1],1/tangs[0]]
            AmP=[nV**2*TV+nW**2*TW,nV**2*TV+nZ**2*TZ,nW**2*TW +nZ**2*TZ]

            #Mean Curvature Normal Factors (H)
            self.NHv[self.EN[e][0]] += -V*TV -W*TW
            self.NHv[self.EN[e][1]] +=  V*TV -Z*TZ
            self.NHv[self.EN[e][2]] +=  W*TW +Z*TZ

            #Assign Node Normal and Node Areas
            for nei,n in enumerate(self.EN[e]):
                self.NARing[n]+=self.EAreas[e]
                self.NAreaG[n]-=self.EAngs[e,nei]*(1+self.NIsB[n])
                if not np.any(IsObtuse):
                    self.NAmix[n]+=AmP[nei]/8
                elif IsObtuse[nei]:
                    self.NAmix[n]+=self.EAreas[e]/2
                else:
                    self.NAmix[n]+=self.EAreas[e]/4
        
        # Dihedral Angles
        for si in range(self.NSides):
            if self.Sides['ef'][si] != -1:
                ei=self.Sides['ei'][si]
                ef=self.Sides['ef'][si]
                Normi=self.ENorms[ei]
                Normf=self.ENorms[ef]
                PCi=self.GetCentroid(ei)
                PCf=self.GetCentroid(ef)
                beta = np.sign(np.dot(Normi,PCi-PCf))*np.arccos(np.dot(Normi,Normf))
                
                PI,PF = [self.Nodes.Mat[self.Sides[lbl][si]] for lbl in ['ni','nf']] 
                PI,PF = [np.array([P['x'],P['y'],P['z']]) for P in [PI,PF]]
                EE=PF-PI
                nEE=np.linalg.norm(EE)
                S = np.outer(EE,EE)*beta/nEE

                self.Sides['beta'][si] = beta
                self.Sides['nEE'][si] = nEE
                self.Sides['S'][si] = S
                
        #Final Nodal Operations
        for n in range(self.NNodes):
            self.NHv[n]/=(4*self.NAmix[n]) # Mean Curvature Normal Vector
            self.NH[n]=np.linalg.norm(self.NHv[n,:])*np.sign(np.dot(self.NHv[n,:],self.NNorms[n])) #Mean Curvature
            self.NK[n]=self.NAreaG[n]/self.NAmix[n] # Gaussian Curvature
            for si in self.NS[n]:
                self.NSdihe[n]+=self.Sides['S'][si]
                self.NHdihe[n]+=self.Sides['beta'][si]*self.Sides['nEE'][si]
            self.NSdihe[n]/=(2*self.NAmix[n]) # Structure Tensor
            self.NHdihe[n]/=(4*self.NAmix[n]) # Mean Curvature
            
        self.NSdiheS=self.NSdihe.copy()  
            
    @staticmethod
    def IndividualCurvPrincipalDirections(ShapeOperator,NHv):
        '''Function to compute principal directions from Shape Operator. It is
        a static function and independent of the actual mesh. Can be used externally.

        Input:
            ShapeOperator - 3x3 array of the Shape Operator
            NHv - array with Mean Curvature Normal Vector 
        Output:
            Vmax - Principal direction vector of principal curvature with maximum magnitude
            Vmin - Principal direction vector of principal curvature with minimum magnitude
            Vnor - Normal of the surface as given by the eigenvector of with zero eigenvalue (not as good)
            kmax - Principal curvature with maximum magnitude
            kmin - Principal curvature with minimum magnitude
            Nktype - type of curvature point (0- both positive, 1- both negative, 2- hyperbolic paraboloid)
            alph - angle of direction of zero curvature (or min mag) from max princ curv direction
            VMM1 - first direction of zero curvature (or min mag)
            VMM2 - second direction of zero curvature (or min mag)
        '''

        minmax=np.array([[1,2],[0,2],[0,1]]) # map position of zero eval to intexes of minimum and maximum eval
        eigs,eigvs= np.linalg.eigh(ShapeOperator) # compute eignevalues (eval) and eigenvectors (evec)
        imaxC=np.argmax(np.abs(eigs)) # compute index of maximum eignevalue magnitude
        izero=np.argmin(np.abs(eigs)) # compute index of zero eigenvalue (for the normal)
        imin,imax=minmax[izero] # get indexes of min and max eval
        
        if imaxC==imin: # minimum eval has the maximum magnitude
            Vmax=eigvs[:,imin]
            Vmin=np.cross(Vmax,NHv)
            Vmin/=np.linalg.norm(Vmin)
        else: # maximum eval has maximum magnitude
            Vmin=eigvs[:,imax]
            Vmax=np.cross(Vmin,NHv)
            Vmax/=np.linalg.norm(Vmax)
        Vnor = eigvs[:,izero]
        # self.NMinCd[n] = eigvs[:,imaxC]
        # self.NMaxCd[n] = Vmax
        # self.NMinCd[n] = Vmin
        # self.NNorCd[n] = eigvs[:,izero]
        
        kmin,kmax=eigs[[imin,imax]] # curvature values
        # self.Nkmin[n]=kmin
        # self.Nkmax[n]=kmax
        # self.NH[n] == (kmin+kmax)/2
        # self.NK[n] == (kmin*kmax)
        
        if kmin>0: # all evals are positive
            Nktype=0
            alph=np.pi/2
            VMM1=Vmin
            VMM2=Vmin
        elif kmax<0: # all evals are negative
            Nktype=1
            alph=0.0
            VMM1=Vmax
            VMM2=Vmax
        else: #kmin<0 and kmax>0 - Hyperbolic paraboloid
            Nktype=2
            alph = np.arctan2(np.sqrt(kmax),np.sqrt(-kmin))
            # print(alph*180/3.1415)
            VMM1=np.cos(alph)*Vmax+np.sin(alph)*Vmin
            VMM2=np.cos(alph)*Vmax-np.sin(alph)*Vmin
        VMM1/=np.linalg.norm(VMM1)
        VMM2/=np.linalg.norm(VMM2)
        
        return Vmax,Vmin,Vnor,kmax,kmin,Nktype,alph,VMM1,VMM2

    def ComputeCurvaturePrincipalDirections(self,usesmooth=True):
        '''Compute principal direction of curvature and get also directions of zero curvature.'''

        self.NMaxCd = np.zeros((self.NNodes,3),dtype='f8') #PD of max curvature (from diheadral)
        self.NMinCd = np.zeros((self.NNodes,3),dtype='f8') #PD of min curvature (from diheadral)
        self.NNorCd = np.zeros((self.NNodes,3),dtype='f8') #PD of normal (from diheadral)
        self.NMinMag= np.zeros((self.NNodes,3),dtype='f8') #PD of min magnitude curvature (from diheadral)  
        self.Nkmin  = np.zeros(self.NNodes,dtype='f8') #k of min curvature (from diheadral)
        self.Nkmax  = np.zeros(self.NNodes,dtype='f8') #k of max curvature (from diheadral) 
        self.Nktype = np.zeros(self.NNodes,dtype='u1') #type of S (0: only positive, 1: only negative, 2: mixed)
        self.Nkang  = np.zeros(self.NNodes,dtype='f8') #angle of zero/min curvature
        self.NMM1= np.zeros((self.NNodes,3),dtype='f8') #PD of min magnitude curvature (from diheadral)  
        self.NMM2= np.zeros((self.NNodes,3),dtype='f8') #PD of min magnitude curvature (from diheadral)  
        
        
        if usesmooth:
            ShapeOperator=self.NSdiheS
        else:
            ShapeOperator=self.NSdihe
        
        for n in range(self.NNodes):
            Vmax,Vmin,Vnor,kmax,kmin,Nktype,alph,VMM1,VMM2 = MyMesh.IndividualCurvPrincipalDirections(ShapeOperator[n],self.NHv[n]) 
            # if n==268:
            #     ShapeOperator[n][0,1]=999
            self.NMaxCd[n] = Vmax
            self.NMinCd[n] = Vmin
            self.NNorCd[n] = Vnor
            self.Nkmin[n]=kmin
            self.Nkmax[n]=kmax
            self.Nktype[n]=Nktype
            self.Nkang[n]=alph
            self.NMinMag[n]=VMM1
            self.NMM1[n]=VMM1
            self.NMM2[n]=VMM2
            
    def SmoothCurvatures(self,N=1):
        '''Smooth the curvatures my averaging the shape operators'''
        for _ in range(N): #Smooth N times
            source=self.NSdiheS.copy()
            for n in range(self.NNodes):
                isb=self.NIsB[n]
                S=source[n]
                if isb:
                    S*=0
                Sn=S*0
                Nns=0
                for n1 in self.NN[n]:
                    if not self.NIsB[n1]:
                        Nns+=1
                        Sn+=source[n1]
                if Nns>0: Sn/=Nns 
                self.NSdiheS[n]=0.5*(S+Sn) #SHOULD BE S+Sn
    
    def ReadjustMinCurvDirections(self,Director_Vector=None,Auto_Director_Vector_Type=0,Make_Zero_Boundary=True,Aux_Director_Vector=None,Director_Vector_List=None):
        '''
            Function to readjust the vectors of minimum curvature. 

            If first finds which of the two directions of zero curvature is more aligned with the director vector.
            The director vector can be: a) given directly by passing it through "Director_Vector"; b) computed
            automatically by setting the algorithm in "Auto_Director_Vector_Type"; or c) computed based on
            "Director_Vector_List" which contains points and directions.

            The sign of the chosed direction of zero curvature vector is then adapted to be in the same direction as
            the auxiliary director vector. This can be set directly through "Aux_Director_Vector" or assumed to be the same
            as the director vector.

            Input:
                Director_Vector (default:None) -- Vector that determines the direction 
                Auto_Director_Vector_Type (default:0) -- Automatic director vector based on local or global average direction. 
                Make_Zero_Boundary (default: True) -- No vector on boundary
                Aux_Director_Vector (default:None) -- Vector that determines the orientation after aligning with the Director_Vector
                Director_Vector_List (default:None) -- List of Lists of vectors [[P1,V1],[P2,V2],...] for precise orientation

            Values for "Auto_Director_Vector_Type". Automatic Computation of Director Vector is based on:
                0 -- GLOBAL Average direction
                1 -- 001 - Average direction of        +        + Node
                2 -- 010 - Average direction of        + 1-Ring +       
                3 -- 011 - Average direction of        + 1-Ring + Node
                4 -- 100 - Average direction of 2-Ring +        +       
                5 -- 101 - Average direction of 2-Ring +        + Node
                6 -- 110 - Average direction of 2-Ring + 1-Ring +       
                7 -- 111 - Average direction of 2-Ring + 1-Ring + Node
        '''
        Compute_Local_Average=False
        Flag_Aux_Director_Vector=False
        
        if Aux_Director_Vector is not None: # Aux Director Vector is external
            Aux_Director_Vector=np.array(Aux_Director_Vector)
            Flag_Aux_Director_Vector=True

        if Director_Vector is not None: # Convert Director Vector
            Director_Vector=np.array(Director_Vector)

        elif Director_Vector_List is not None: # Set-up arrays for List of Director Vectors
            N_dpoints = len(Director_Vector_List)
            DirVecPoints = np.zeros((N_dpoints,3),dtype='f8')
            DirVecVectors = np.zeros((N_dpoints,3),dtype='f8')
            for i in range(len(Director_Vector_List)):
                DirVecPoints[i,:]=np.array(Director_Vector_List[i][0])
                DirVecVectors[i,:]=np.array(Director_Vector_List[i][1])

        else: # Automatic Computation of Director Vector
            if Auto_Director_Vector_Type == 0:
                Director_Vector=np.zeros(3)
                for n in range(self.NNodes):
                    if not self.NIsB[n]:
                        Director_Vector+=self.NMM1[n]
                Director_Vector/=np.linalg.norm(Director_Vector)
            elif Auto_Director_Vector_Type > 7:
                raise ValueError(" Auto_Director_Vector_Type has undefined value ("+str(Auto_Director_Vector_Type)+")")
            else:
                Compute_Local_Average=True


        VTemp=np.zeros_like(self.NMM1)
        for n in range(self.NNodes): # Go node-by-node to fix vector
            if not (self.NIsB[n] and Make_Zero_Boundary): # Make boundary vectors 0 if desired
                
                # Compute special automatic director vector
                if Compute_Local_Average:
                    Node_set=set()
                    if Auto_Director_Vector_Type in [4,5,6,7]:
                        # Compute N2+N1+n
                        for n1 in self.NN[n]:
                            if not self.NIsB[n1]:
                                for n2 in self.NN[n1]:
                                    if not self.NIsB[n2]:
                                        Node_set.add(n2)
                        if Auto_Director_Vector_Type in [4,5]:
                            Node_set.difference_update(self.NN[n]) # Remove N1
                        if Auto_Director_Vector_Type in [4,6]:
                            Node_set.discard(n) # Remove Node
                    else:
                        if Auto_Director_Vector_Type in [2,3]:
                            for n1 in self.NN[n]:
                                if not self.NIsB[n1]:
                                    Node_set.add(n1) # Add N1
                        if Auto_Director_Vector_Type in [1,3]:
                            Node_set.add(n) #Add Node
                    Director_Vector=np.zeros(3)
                    for ns in Node_set:
                        Director_Vector+=self.NMM1[ns]
                    Director_Vector/=np.linalg.norm(Director_Vector)

                # Set director vector from list
                if Director_Vector_List is not None:
                    # Find closest 3 points (p1,p2,p3) and get director vector as v1 + v2*d1/d2 + v3*d1/d3
                    P=self.Nodes.Mat[n]
                    Pos = np.array([P['x'],P['y'],P['z']])
                    Distances = np.sum((DirVecPoints-Pos)**2,axis=1)**0.5
                    Isort = np.argsort(Distances)
                    d1=Distances[Isort[0]]
                    Director_Vector = DirVecVectors[Isort[0],:].copy()
                    for i in range(1,min(N_dpoints,3)):
                        Director_Vector += DirVecVectors[Isort[i],:] * d1/Distances[Isort[i]]
                    Director_Vector/=np.linalg.norm(Director_Vector)

                    
                Va=self.NMM1[n] # Get vector of min magnitude
                dot_Va_DirVec=np.dot(Director_Vector,Va) # Got dot product with director vector
                dot_Va_AuxVec=np.dot(Aux_Director_Vector,Va) if Flag_Aux_Director_Vector else dot_Va_DirVec
                if dot_Va_AuxVec<0: Va*=-1 #Set vector to correct direction
                VTemp[n]=Va # Fix direction 
                
                if self.Nktype[n]==2: # If it has mixed curvature, i.e. pair of orientations with 0 curvature
                    # Compute second zero curvature vector
                    Vb=self.NMM2[n]
                    Vb/=np.linalg.norm(Vb)
                    dot_Vb_DirVec=np.dot(Director_Vector,Vb)
                    # Set second if it closer to director vector
                    if abs(dot_Va_DirVec)<abs(dot_Vb_DirVec):
                        dot_Vb_AuxVec=np.dot(Aux_Director_Vector,Vb) if Flag_Aux_Director_Vector else dot_Vb_DirVec
                        if dot_Vb_AuxVec<0: Vb*=-1
                        VTemp[n]=Vb

                # Test Director Vector
                # VTemp[n]=Director_Vector

        self.NMinMag=VTemp

    def FlipMinCurvDirections(self,Aux_Director_Vector=None):
        '''
        Flip to the other direction of zero curvature based on current direction of zero curvature.
        Useful for plotting both streamlines.
        '''
        VTemp=np.zeros_like(self.NMM1)
        isnotzero= np.logical_not(np.all(self.NMinMag == VTemp,axis=1))
        isMM1 = np.logical_and(isnotzero,np.all(self.NMinMag == self.NMM1,axis=1))
        isMM2 = np.logical_and(isnotzero,np.logical_not(isMM1))
        VTemp[isMM1]=self.NMM2[isMM1]
        VTemp[isMM2]=self.NMM1[isMM2]

        if Aux_Director_Vector is not None:
            Aux_Director_Vector=np.array(Aux_Director_Vector)
            for n in range(self.NNodes): # Go node-by-node to fix vector
                if isnotzero[n]:
                    Va=VTemp[n] # Get vector of min magnitude
                    dot_Va_AuxVec=np.dot(Aux_Director_Vector,Va)
                    if dot_Va_AuxVec<0: Va*=-1 #Set vector to correct direction
                    VTemp[n]=Va # Fix direction 

        self.NMinMag=VTemp

    def GetAllFieldsForInterpolation(self):
        return [plbl for plbl in self.Nodes.PropTypes.Ltype if not plbl == 'N']

    def InterpolateFieldForPoint(self,point,field_lbl,DoZ=False):
        '''
        Given a point x,y(,z) finds the closest point in mesh surface and computes
        the interpolated value of the requested field at that position.

        point - x,y(,z) coordinates to interpolate for.
        field_lbl - label of field to interpolate. If 'all' then all fields present are interpolated.
        DoZ[False] - If true also checks the z coordinate (useful is mesh is not unique in x,y)

        '''

        ret_values=[]

        # Make label list 
        if type(field_lbl) == type([]):
            fieldLabels = field_lbl
        elif field_lbl.lower() =='all': # Get all data
            fieldLabels = self.GetAllFieldsForInterpolation()
        else:
            fieldLabels = [field_lbl]

        # Get interpolation properties
        nodes,baris = self._getInterpolatingProps(point,DoZ=DoZ)

        # Collect and interpolate all fields
        for plbl in fieldLabels:
            #ret_labels.append(plbl)
            ret_values.append(self._quickInterp(plbl,nodes,baris))

        #return ret_labels,ret_values
        return ret_values

    @staticmethod
    def _getBaricentricCoordinates(point,P1,P2,P3):
        return _Aux_getBaricentricCoordinates(point,P1,P2,P3)


    def _quickInterp(self,field_lbl,nodes,baris):
        val=0.0
        for node,bari in zip(nodes,baris):
            val+=self.Nodes.Mat[field_lbl][node]*bari

        return val

    def _getInterpolatingProps(self,point,DoZ=False):
        XX = point[0]
        YY = point[1]
        if DoZ: ZZ = point[2]

        #Get Nodes X,Y,Z
        NX=self.Nodes.Mat['x']
        NY=self.Nodes.Mat['y']
        NZ=self.Nodes.Mat['z']

        #Get distance squared from point to nodes
        dNodes = (NX-XX)**2+(NY-YY)**2
        if DoZ: dNodes+=(NZ-ZZ)**2
        SortedIndex = np.argsort(dNodes)

        # Closest node relative to median distance
        relative_first_node_distance = dNodes[SortedIndex[0]]/dNodes[SortedIndex[int(len(dNodes)/2)]] #d(0)/d(L/2)

        if  relative_first_node_distance < 1E-10: #Distance very small compared to Median distance
            return [SortedIndex[0]],[1.0]

        else:
            #Find up to 15 nodes close to point
            ReasonableSize=min(len(dNodes),15)
            CloseNodes=SortedIndex[:ReasonableSize]
            CloseElements = []
            for el_list in self.NE[CloseNodes]:
                CloseElements.extend(el_list)
            CloseElements=np.unique(CloseElements)
            
            for e in CloseElements:
                nodes = self.EN[e]
                P1,P2,P3 = [np.array([P['x'],P['y']]) for P in self.Nodes.Mat[nodes]]
                l1,l2,l3 = self._getBaricentricCoordinates(np.array([XX,YY]),P1,P2,P3)
                if (l1>=0 and l2>=0 and l3>=0 and l1<=1 and l2<=1 and l3<=1): #Is inside Triangle
                    return nodes,[l1,l2,l3]
                         

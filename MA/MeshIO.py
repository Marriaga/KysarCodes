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
import xml.dom.minidom


# Get Endienness of System
SYS_ENDI = {'little': '<','big': '>'}[sys.byteorder]

# Dictionary for ply->numpy endienness
ENDI_CONVERSION = {
'ascii': '=',
'binary_little_endian': '<',
'binary_big_endian': '>'}
# Dictionary for numpy->ply endienness
INV_ENDI_CONVERSION = MATL.invidct(ENDI_CONVERSION)

# Get Endi in numpy format (from ply)
def GetEndinp(val):
    return ENDI_CONVERSION[val]    

# Get Endi in ply format (from numpy)
def GetEndiply(val):
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

# Get datatype in numpy format (from ply)
def ply2numpy(plytype,endi):
    #endi: < - little; > - Big
    if endi == "=": endi = SYS_ENDI
    return endi + TYPE_CONVERSION[plytype]
    
# Get datatype in ply format (from numpy)
def numpy2ply(nptype):
    if nptype[0] in "=<>": nptype = nptype[1:]
    return INV_TYPE_CONVERSION[nptype]

# Class for collection of properties    
class MyPropTypes(object):
    def __init__(self):
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
        
    # Changes or adds a type in the available types
    def AddBaseType(self,typlbl,type=None):
        if type is None:
            type='f8'
            print("Warning: Added label "+typlbl+" with no associated type. 'double' assumed")
        self.AllTypes[typlbl] = type
    
    # Adds a type to the type list    
    def AddTypeToList(self,typlbl,type=None):
        if (typlbl not in self.AllTypes) or (type is not None):
            self.AddBaseType(typlbl,type)
        
        if typlbl not in self.Ltype:
            self.Ltype.append(typlbl)
            
    # Get the type of a field with a particular label
    def GetType(self,typlbl):
        if typlbl not in self.AllTypes:
            self.AddBaseType(typlbl)
        return self.AllTypes[typlbl]
    
    # Normalize Endienness
    def NormEndi(self,Endi):
        for typlbl in self.Ltype:
            typtyp = self.GetType(typlbl)
            if typtyp[0] in "=<>":
                typtyp = typtyp[1:]
            typtyp=Endi+typtyp
            self.AddBaseType(typlbl,typtyp)
    
    # Get Numpy dtype data
    def GetDtype(self):           
        ListDTypes=[]
        for typlbl in self.Ltype:
            ListDTypes.append((typlbl,self.GetType(typlbl)))
        return ListDTypes
    
    # Get NProp
    def GetNProp(self):
        return len(self.Ltype)
        
    #Generates the List
    def MakeListType(self,Types,append=False):
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
        self.is_sequencial=False # Nodes are sequencial in numbering
        self.si=0      # Starting index
        self.Mat=[]    # Numpy array with values
        self.PropTypes = MyPropTypes() # Data types
        
        if Matrix is not None:
            self.PutMatrix(Matrix,**kwargs)
        
    def GetNNodes(self):
        return len(self.Mat)
    
    def AppendField(self,myarray,lbl,atype=None):
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
    
    # Takes the Matrix array and creates the Numpy array with the correct data type
    def PutMatrix(self,Matrix,Types=None,isnumbered=False):
        #isnumbered - Flag to indicate that the input matrix is numbered
        Matrix=np.array(Matrix)
        try:
            NN,NF = np.shape(Matrix)
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
            Matrix=nprec.merge_arrays((np.arange(NN,dtype=self.PropTypes.GetType('N')),Matrix), flatten = True)
            #Matrix=np.insert(Matrix,0,np.arange(NN,dtype='i4'),axis=1)
            
        # Add z=0 for 2D Data    
        if (NF == 2) or (NF == 3 and isnumbered): #Assumed 2D Structure
            Matrix=nprec.append_fields(Matrix,'z',np.zeros(NN,dtype=self.PropTypes.GetType('z')))
            #Matrix=np.append(Matrix,np.zeros((NN,1),dtype='f8'),axis=1)
        
        # Save Matrix
        Matrix.dtype.names=typelabs
        self.Mat = np.sort(Matrix,order='N')
        
        # Check ordering and starting item
        MN = self.Mat['N']
        if MN[-1]-MN[0]+1 == len(MN):
            self.is_sequencial=True
            self.si=MN[0]
        
# Class for Elements
class CElements(object):
    def __init__(self,Matrix=None,**kwargs):
        self.is_sequencial=False # Nodes are sequencial in numbering
        self.si=0      # Starting index
        self.Mat=[]    # Numpy array with values
        self.PropTypes = MyPropTypes() # Data types
        
        if Matrix is not None:
            self.PutMatrix(Matrix,**kwargs)
        
    def GetNElems(self):
        return len(self.Mat)
        
    # Takes the Matrix array and creates the Numpy array with the correct data type
    def PutMatrix(self,Matrix,Types=None,isnumbered=False):
        #isnumbered - Flag to indicate that the input matrix is numbered
        Matrix=np.array(Matrix)
        try:
            NN,NF = np.shape(Matrix)
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
            Matrix=nprec.merge_arrays((np.arange(NN,dtype=self.PropTypes.GetType('N')),Matrix), flatten = True)
            #Matrix=np.insert(Matrix,0,np.arange(NN,dtype='i4'),axis=1)
        
        # Save Matrix
        Matrix.dtype.names=typelabs
        self.Mat = np.sort(Matrix,order='N')
        
        # Check ordering and starting item
        MN = self.Mat['N']
        if MN[-1]-MN[0]+1 == len(MN):
            self.is_sequencial=True
            self.si=MN[0]


class BaseIO(object):
    def __init__(self):
        self.PropTypes=MyPropTypes()
        self.EPT=MyPropTypes()
        self.Endi = SYS_ENDI
        self.NNodes = -1
        self.NElem = -1
        self.Nodes = []
        self.Elems = []

    # Reformat data to match Endi
    def NormEndi(self):
        self.PropTypes.NormEndi(self.Endi)
        self.Nodes.Mat=self.Nodes.Mat.astype(self.PropTypes.GetDtype())
        self.EPT.NormEndi(self.Endi)
        self.Elems.Mat=(self.Elems.Mat.astype(self.EPT.GetDtype()))
        
    ## -- DATA INPUT/OUTPUT OPERATIONS -- ##
    
    def ExportMesh(self):
        return self.Nodes, self.Elems
        
    def ImportMesh(self,Nodes,Elems):
        self.Nodes = Nodes
        self.Elems = Elems
        self.RefreshNodes()
        self.RefreshElems()
        
    # Refresh Local Properties for Nodes
    def RefreshNodes(self):
        self.PropTypes = self.Nodes.PropTypes
        self.NNodes = self.Nodes.GetNNodes()
        
    # Refresh Local Properties for Elemens
    def RefreshElems(self):
        self.EPT = self.Elems.PropTypes
        self.NElem = self.Elems.GetNElems()

    def SetNodes(self,Matrix,**kwargs):
        self.Nodes = CNodes(Matrix,**kwargs)
        self.RefreshNodes()
        
    def SetElems(self,Matrix,**kwargs):
        self.Elems = CElements(Matrix)
        self.RefreshElems()
            
# Class for PLY I/O
class PLYIO(BaseIO):
    def __init__(self,Load_File=None):
        BaseIO.__init__(self)
        if Load_File is not None:
            self.LoadFile(Load_File)
    
    ## -- READING PLY OPERATIONS -- ##
    
    # Load File
    def LoadFile(self,plyfile):
        with open(plyfile, 'rb') as fp:
            self.ReadHeader(fp)
            if self.Endi == "=":
                self.ReadNodesASCII(fp)
                self.ReadElemsASCII(fp)
            else:
                self.ReadNodesBinary(fp)
                self.ReadElemsBinary(fp)
                
    # Read the ply header            
    def ReadHeader(self,fp):
        stage=0
        while True:
            lvs=fp.readline().rstrip().split()
            
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
                      
    # Read Node Info in ASCII
    def ReadNodesASCII(self,fp):
        Nodes = np.empty(self.NNodes,dtype=np.dtype(self.PropTypes.GetDtype()))
        NP = self.PropTypes.GetNProp()
        for i in range(self.NNodes):
            linelist=fp.readline().strip().split()
            for p in range(NP):
                Nodes[i][p]=linelist[p]
        self.SetNodes(Nodes,Types=self.PropTypes)
                
    # Read Element Info in ASCII
    def ReadElemsASCII(self,fp):
        Elems = np.zeros(self.NElem,dtype=[('p1','i4'),('p2','i4'),('p3','i4')])
        NP = 3
        for i in range(self.NElem):
            linelist=fp.readline().strip().split()[1:]
            for p in range(NP):
                Elems[i][p]=linelist[p]
        self.SetElems(Elems)

    # Read Node Info in ASCII
    def ReadNodesBinary(self,fp):
        Nodes = np.fromfile(fp,dtype=self.PropTypes.GetDtype(),count=self.NNodes)
        self.SetNodes(Nodes,Types=self.PropTypes)
                
    # Read Element Info in ASCII
    def ReadElemsBinary(self,fp):
        Elems = nprec.drop_fields(np.fromfile(fp,dtype=[('count', 'u1'),('p1','i4'),('p2','i4'),('p3','i4')],count=self.NElem),'count')
        self.SetElems(Elems)
         
        
    ## -- WRITE PLY OPERATIONS -- ##
    
    def SaveFile(self,plyfile,Endi=None,asASCII=False):
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
            
    # Write the ply header
    def WriteHeader(self,fp,asASCII):
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
        
        fp.write(header)

    def WriteArray(self,fp,Array):
        for line in Array:
            fp.write(" ".join([str(el) for el in line]) + "\n")
        
    def WriteNodesASCII(self,fp): 
        NoNLabs=self.PropTypes.Ltype[1:]
        TmpNodes=self.Nodes.Mat[NoNLabs].copy()
        self.WriteArray(fp,TmpNodes)
        
    def WriteElemsASCII(self,fp):    
        tmpdtype = self.Elems.Mat.dtype.descr
        tmpdtype[0]=('N', self.Endi+'u1')
        TmpElems = self.Elems.Mat.astype(tmpdtype)
        TmpElems['N']=3
        self.WriteArray(fp,TmpElems)
        
    def WriteNodesBinary(self,fp):
        NoNLabs=self.PropTypes.Ltype[1:]
        TmpNodes=self.Nodes.Mat[NoNLabs].copy().data
        TmpNodes.tofile(fp)
        
    def WriteElemsBinary(self,fp):
        tmpdtype = self.Elems.Mat.dtype.descr
        tmpdtype[0]=('N', 'u1')
        TmpElems =self.Elems.Mat.astype(tmpdtype)
        TmpElems['N']=3
        TmpElems.tofile(fp)

# Class for VTU I/O        
class VTUIO(BaseIO):
    def __init__(self):
        BaseIO.__init__(self)
        self.VecLabs=[]

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
        if Endi is not None:
            self.Endi = Endi
        if self.Endi == "=":
            self.Endi=SYS_ENDI
        self.NormEndi()
    
        self.SetPiece()
        self.SetNodes()
        self.SetElems()
        
        if self.VecLabs==[]:
            print("VecLabs is empty. Only geometry exported. Do VTUIO.VecLabs.append(['Name',[labs]]) to add data")
        
        self.SetNodeData()
        self.SetElemData()
        
        with open(vtufile, 'wb') as fp:
            self.doc.writexml(fp, newl='\n')
    
    def SetPiece(self):
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
        cell_data = self.doc.createElementNS("VTK", "CellData")
        self.piece.appendChild(cell_data)
        
        

class MyMesh(object):
    def __init__(self,Nodes,Elems):
        self.NNodes = Nodes.GetNNodes()
        self.NElems = Elems.GetNElems()
        self.Nodes  = Nodes
        self.Elems  = Elems
        self.IndxL = self.MakeIndexed()
        
        # Neighbours in 1-Ring
        self.NN = np.empty(self.NNodes,dtype=object)
        self.NE = np.empty(self.NNodes,dtype=object)
        self.EN = np.empty(self.NElems,dtype=object)
        self.EE = np.empty(self.NElems,dtype=object)
        self.ComputeNeighbours()
        
        # Sides in 1-Ring
        #self.Sides=np.zeros(self.NSides,dtype=[('ni','i4'),('nf','i4'),('ei','i4'),('ef','i4'),('beta','f8')])
        self.NS = np.empty(self.NNodes,dtype=object)
        self.ES = np.zeros((self.NElems,3),dtype='i4')
        self.ComputeSides()
        
        # Check if nodes, elements or sides are in boundary of mesh
        self.NIsB = np.zeros(self.NNodes,dtype=np.bool)
        self.EIsB = np.zeros(self.NElems,dtype=np.bool)
        self.SIsB = np.zeros(self.NSides,dtype=np.bool)
        self.ComputeBoundary()

    def MakeIndexed(self):
        self.NewElemsMat=np.zeros((self.NElems,3),dtype='i4')
        # "Allocate empty structured array"
        IndxL=np.empty(self.NNodes,dtype=[('N','i4'),('I','i4')])
        
        # "Fill N and I"
        IndxL['N'] = self.Nodes.Mat['N']
        IndxL['I'] = np.arange(self.NNodes)
        
        # Sort
        IndxL.sort(order='N')
        
        # Add Ip1, Ip2 and Ip3
        idxp1 = IndxL['N'].searchsorted(self.Elems.Mat['p1'])
        idxp2 = IndxL['N'].searchsorted(self.Elems.Mat['p2'])
        idxp3 = IndxL['N'].searchsorted(self.Elems.Mat['p3'])
        self.NewElemsMat[:,0] = IndxL['I'][idxp1]
        self.NewElemsMat[:,1] = IndxL['I'][idxp2]
        self.NewElemsMat[:,2] = IndxL['I'][idxp3]

        return IndxL
    
    def ComputeNeighbours(self):
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
        if n1==n2: raise ValueError("Cannot find side of same node")
        for val in self.NS[n1]:
            if val in self.NS[n2]:
                return val
        raise ValueError("Cannot find side for node pair")
            
    def ComputeBoundary(self):
        # Compute EIsB and NIsB
        for si in range(self.NSides):
            ni,nf,ei,ef = [self.Sides[si][lab] for lab in ['ni','nf','ei','ef']]
            if ef==-1:
                self.NIsB[ni]=True
                self.NIsB[nf]=True
                self.EIsB[ei]=True
                self.SIsB[si]=True
                
    def ComputeNormals(self,mode=0):
        #Mode: Type of Nodal normal computation 
        #  0-Simple Average
        #  1-Weighted by Area
        #  2-Weighted by Angle
        if mode==0:
            def mymode(e,nei): return 1.0
        if mode==1:
            def mymode(e,nei): return self.EAreas[e]
        if mode==2:
            def mymode(e,nei): return self.EAngs[e,nei]

        self.ENorms = np.empty((self.NElems,3),dtype='f8') #nx,ny,nz
        self.EAngs  = np.empty((self.NElems,3),dtype='f8') #a1,a2,a3
        self.EAreas = np.empty(self.NElems,dtype='f8') #A
        self.NNorms = np.zeros((self.NNodes,3),dtype='f8') #nx,ny,nz
        
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
        C=np.zeros(3)
        for P in self.Nodes.Mat[self.EN[e]]:
            C+=[P['x'],P['y'],P['z']]
        return C/3.0    

    
    def ComputeCurvatures(self):    
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
            
    def ComputeCurvaturePrincipalDirections(self,usesmooth=True):
        self.NMaxCd = np.zeros((self.NNodes,3),dtype='f8') #PD of max curvature (from diheadral)
        self.NMinCd = np.zeros((self.NNodes,3),dtype='f8') #PD of min curvature (from diheadral)
        self.NNorCd = np.zeros((self.NNodes,3),dtype='f8') #PD of normal (from diheadral)
        self.NMinMag= np.zeros((self.NNodes,3),dtype='f8') #PD of min magnitude curvature (from diheadral)  
        self.Nkmin  = np.zeros(self.NNodes,dtype='f8') #k of min curvature (from diheadral)
        self.Nkmax  = np.zeros(self.NNodes,dtype='f8') #k of max curvature (from diheadral) 
        self.Nktype = np.zeros(self.NNodes,dtype='u1') #type of S (0: only positive, 1: only negative, 2: mixed)
        self.Nkang  = np.zeros(self.NNodes,dtype='f8') #angle of zero/min curvature
        
        
        if usesmooth:
            ShapeOperator=self.NSdiheS
        else:
            ShapeOperator=self.NSdihe
        
        minmax=[[1,2],[0,2],[0,1]]
        for n in range(self.NNodes):
            eigs,eigvs= np.linalg.eigh(ShapeOperator[n])
            imaxC=np.argmax(np.abs(eigs))
            izero=np.argmin(np.abs(eigs))
            imin,imax=minmax[izero]
            
            if imaxC==imin:
                Vmax=eigvs[:,imin]
                Vmin=np.cross(Vmax,self.NHv[n])
                Vmin/=np.linalg.norm(Vmin)
            else:
                Vmin=eigvs[:,imax]
                Vmax=np.cross(Vmin,self.NHv[n])
                Vmax/=np.linalg.norm(Vmax)
            
            # self.NMinCd[n] = eigvs[:,imaxC]
            self.NMaxCd[n] = Vmax
            self.NMinCd[n] = Vmin
            self.NNorCd[n] = eigvs[:,izero]
            
            kmin,kmax=eigs[[imin,imax]]
            self.Nkmin[n]=kmin
            self.Nkmax[n]=kmax
            # self.NH[n] == (kmin+kmax)/2
            # self.NK[n] == (kmin*kmax)
            
            if kmin>0:
                self.Nktype[n]=0
                self.Nkang[n]=np.pi/2
                V=Vmin
            elif kmax<0:
                self.Nktype[n]=1
                self.Nkang[n]=0.0
                V=Vmax
            else: #kmin<0 and kmax>0
                self.Nktype[n]=2
                alph = np.arctan2(np.sqrt(kmax),np.sqrt(-kmin))
                self.Nkang[n] = alph
                # print(alph*180/3.1415)
                V=np.cos(alph)*Vmax+np.sin(alph)*Vmin
            V/=np.linalg.norm(V)

            self.NMinMag[n]=V
            
    def SmoothCurvatures(self,N=1):
        for i in range(N): #Smooth N times
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
    
    def ReadjustMinCurvDirections(self,Director_Vector=None,Auto_Director_Vector_Type=0,Make_Zero_Boundary=True,Aux_Director_Vector=None):
        VTemp=np.zeros_like(self.NMinMag)
        Compute_Local_Average=False
        Flag_Aux_Director_Vector=False
        
        if Aux_Director_Vector is not None:
            Aux_Director_Vector=np.array(Aux_Director_Vector)
            Flag_Aux_Director_Vector=True

        if Director_Vector is not None:
            Director_Vector=np.array(Director_Vector)
        else: # Automatic Computation of Director Vector
              # 0 - GLOBAL Average direction
              # 1 - 001 - Average direction of        +        + Node
              # 2 - 010 - Average direction of        + 1-Ring +       
              # 3 - 011 - Average direction of        + 1-Ring + Node
              # 4 - 100 - Average direction of 2-Ring +        +       
              # 5 - 101 - Average direction of 2-Ring +        + Node
              # 6 - 110 - Average direction of 2-Ring + 1-Ring +       
              # 7 - 111 - Average direction of 2-Ring + 1-Ring + Node

            if Auto_Director_Vector_Type == 0:
                Director_Vector=np.zeros(3)
                for n in range(self.NNodes):
                    if not self.NIsB[n]:
                        Director_Vector+=self.NMinMag[n]
                Director_Vector/=np.linalg.norm(Director_Vector)
            elif Auto_Director_Vector_Type > 7:
                raise ValueError(" Auto_Director_Vector_Type has undefined value ("+str(Auto_Director_Vector_Type)+")")
            else:
                Compute_Local_Average=True

        for n in range(self.NNodes): # Go node-by-node to fix vector
            if not (self.NIsB[n] and Make_Zero_Boundary): #Make boundary vectors 0 if desired
                
                if Compute_Local_Average: #Get Local Director Vector
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
                        Director_Vector+=self.NMinMag[ns]
                    Director_Vector/=np.linalg.norm(Director_Vector)
                
                
                Va=self.NMinMag[n] # Get vector of min magnitude
                dot_Va_DirVec=np.dot(Director_Vector,Va) # Got dot product with director vector
                dot_Va_AuxVec=np.dot(Aux_Director_Vector,Va) if Flag_Aux_Director_Vector else dot_Va_DirVec
                if dot_Va_AuxVec<0: Va*=-1 #Set vector to correct direction
                VTemp[n]=Va # Fix direction 
                
                if self.Nktype[n]==2: # If it has mixed curvature, i.e. pair of orientations with 0 curvature
                    # Compute second zero curvature vector
                    Vb=np.cos(self.Nkang[n])*self.NMaxCd[n]-np.sin(self.Nkang[n])*self.NMinCd[n]
                    Vb/=np.linalg.norm(Vb)
                    dot_Vb_DirVec=np.dot(Director_Vector,Vb)
                    # Set second if it closer to director vector
                    if abs(dot_Va_DirVec)<abs(dot_Vb_DirVec):
                        dot_Vb_AuxVec=np.dot(Aux_Director_Vector,Vb) if Flag_Aux_Director_Vector else dot_Vb_DirVec
                        if dot_Vb_AuxVec<0: Vb*=-1
                        VTemp[n]=Vb

        self.NMinMag=VTemp         
            

            
            
# def LoadFile(self,FilePath):
    # path,ext = os.path.splitext(FilePath)
    # if ext == ".ply":
        # Ply = MAMIO.PLYIO(FilePath)
        # Nodes,Elems = Ply.ExportMesh()    
        # MyM = MAMIO.MyMesh(Nodes,Elems)
        # del Ply
    # else:
        # raise ValueError("Loading for that filetype does not exist ("+ext+")"

            
        
# if __name__ == "__main__":
    # print(" No testing model provided")
    # # fn="Tests/CSV3D/test"
    # # MakeCSVMesh(fn+".csv",ColFile=fn+".png",Expfile=fn+".ply")
    
    
    
    
    
    
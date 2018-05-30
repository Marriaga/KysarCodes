#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import sys
import os
#import MA.ImageProcessing as MAIP
import MA.Tools as MATL
import MA.CSV3D as MA3D
import MA.MeshIO as MAMIO

sys.path.append('C:\\Program Files\\VCG\\MeshLab') # MeshLab


def MakeSmoothMeshlabScript(FilePath):
    ''' Make the meshlab script that smooths the mesh'''
    with open(FilePath, "w") as fp:
        fp.write(
        r"""
        <!DOCTYPE FilterScript>
        <FilterScript>
         <filter name="Taubin Smooth">
          <Param value="0.5"  type="RichFloat" description="Lambda" name="lambda"/>
          <Param value="-0.53"  type="RichFloat" description="mu" name="mu"/>
          <Param value="10"  type="RichInt" description="Smoothing steps" name="stepSmoothNum"/>
          <Param value="false"  type="RichBool" description="Affect only selected faces" name="Selected"/>
         </filter>
         <filter name="Simplification: Quadric Edge Collapse Decimation">
          <Param value="1000"  type="RichInt" description="Target number of faces" name="TargetFaceNum"/>
          <Param value="0"  type="RichFloat" description="Percentage reduction (0..1)" name="TargetPerc"/>
          <Param value="1"  type="RichFloat" description="Quality threshold" name="QualityThr"/>
          <Param value="false"  type="RichBool" description="Preserve Boundary of the mesh" name="PreserveBoundary"/>
          <Param value="1"  type="RichFloat" description="Boundary Preserving Weight" name="BoundaryWeight"/>
          <Param value="false"  type="RichBool" description="Preserve Normal" name="PreserveNormal"/>
          <Param value="false"  type="RichBool" description="Preserve Topology" name="PreserveTopology"/>
          <Param value="true"  type="RichBool" description="Optimal position of simplified vertices" name="OptimalPlacement"/>
          <Param value="false"  type="RichBool" description="Planar Simplification" name="PlanarQuadric"/>
          <Param value="false"  type="RichBool" description="Weighted Simplification" name="QualityWeight"/>
          <Param value="true"  type="RichBool" description="Post-simplification cleaning" name="AutoClean"/>
          <Param value="false"  type="RichBool" description="Simplify only selected faces" name="Selected"/>
         </filter>
         <filter name="Taubin Smooth">
          <Param value="0.5"  type="RichFloat" description="Lambda" name="lambda"/>
          <Param value="-0.53"  type="RichFloat" description="mu" name="mu"/>
          <Param value="10"  type="RichInt" description="Smoothing steps" name="stepSmoothNum"/>
          <Param value="false"  type="RichBool" description="Affect only selected faces" name="Selected"/>
         </filter>
         <filter name="Subdivision Surfaces: Loop">
          <Param enum_cardinality="3" enum_val1="Enhance regularity" enum_val0="Loop" value="1" enum_val2="Enhance continuity"  type="RichEnum" description="Weighting scheme" name="LoopWeight"/>
          <Param value="3"  type="RichInt" description="Iterations" name="Iterations"/>
          <Param min="0" value="50"  type="RichAbsPerc" description="Edge Threshold" max="100.0" name="Threshold"/>
          <Param value="false"  type="RichBool" description="Affect only selected faces" name="Selected"/>
         </filter>
         <filter name="Taubin Smooth">
          <Param value="0.5"  type="RichFloat" description="Lambda" name="lambda"/>
          <Param value="-0.53"  type="RichFloat" description="mu" name="mu"/>
          <Param value="10"  type="RichInt" description="Smoothing steps" name="stepSmoothNum"/>
          <Param value="false"  type="RichBool" description="Affect only selected faces" name="Selected"/>
         </filter>
        </FilterScript>
        """)

def SmoothPly(plyInput,plySmooth,ScriptFileName=None,Verbose=False):
    if ScriptFileName is None:
        ScriptFileName=os.path.join(os.path.dirname(plyInput),"SmoothScript")
        MakeSmoothMeshlabScript(ScriptFileName)
    MATL.RunMeshLabScript(plyInput,ScriptFileName,Output_Path=plySmooth,Verbose=Verbose)

def GetVectXYAng(Vec):
    X=Vec[:,0]
    Y=Vec[:,1]
    Ang = np.arctan2(Y,X) * 180/np.pi
    Ang[Ang<0]+=180
    return Ang

def MakePLYWithCurvatureInfo(plyFile_Smooth,plyFile_Curvs,SmoothN):
    Ply = MAMIO.PLYIO()
    print("  Loading Ply File...")
    Ply.LoadFile(plyFile_Smooth)
    print( "  Loading Mesh Data ...")
    Nodes,Elems = Ply.ExportMesh()
    MyM = MAMIO.MyMesh(Nodes,Elems)
    print( "  Computing Normals ...")
    MyM.ComputeNormals()
    print( "  Computing Curvatures ...")
    MyM.ComputeCurvatures()
    print( "  Smoothing Curvatures ...")
    MyM.SmoothCurvatures(SmoothN)    
    print( "  Compute Principal Directions ...")
    MyM.ComputeCurvaturePrincipalDirections()
    print( "  Readjust Min Curvature Directions ...")
    MyM.ReadjustMinCurvDirections()
    print( "  Creating Export Data ...")
    MyM.Nodes.AppendField(MyM.NMinMag[:,0],'nx')
    MyM.Nodes.AppendField(MyM.NMinMag[:,1],'ny')
    MyM.Nodes.AppendField(MyM.NMinMag[:,2],'nz')
    MyM.Nodes.AppendField(GetVectXYAng(MyM.NMaxCd),'Amax')
    MyM.Nodes.AppendField(GetVectXYAng(MyM.NMinCd),'Amin')
    MyM.Nodes.AppendField(GetVectXYAng(MyM.NMinMag),'Amag')
    #MyM.Nodes.AppendField(MyM.Nktype,'T',atype=MyM.Nodes.PropTypes.AllTypes["T"])
    MyM.Nodes.AppendField(MyM.Nktype,'T')
    print( "  Importing Data PLY ...")
    Ply.ImportMesh(MyM.Nodes,MyM.Elems)
    print( "  Saving Data Ply ...")
    Ply.SaveFile(plyFile_Curvs)

def InterpolatePropertiesFromPLY(plyFile_Curvs,PointsList):
    #Compute properties of specific points
    Ply = MAMIO.PLYIO()
    Ply.LoadFile(plyFile_Curvs)
    Nodes,Elems = Ply.ExportMesh()
    MyM = MAMIO.MyMesh(Nodes,Elems)
    Header = MyM.GetAllFieldsForInterpolation()
    properties=[]
    for p in PointsList:
        properties.append(MyM.InterpolateFieldForPoint(p,"All"))
    return Header,properties
            
def MakeAllFromTif(tiffile,PointsList=None):
    Base = os.path.splitext(tiffile)[0]
    plyFile = Base+"_PLY0_Original.ply"
    plyFile_Smooth = Base+"_PLY1_Reduced.ply"
    plyFile_Curvs = Base+"_PLY2_Curvatures.ply"
    txtFile_DataPoints = Base+"_Interpolated_Data_Points"

    # Make ply from height map if heightmap is newer
    if MATL.IsNew(tiffile,plyFile): 
        print("Making 3D Surface...")
        MA3D.Make3DSurfaceFromHeightMapTiff(tiffile,OFile=plyFile,NoZeros=True)

    # Smooth ply file if input ply is new
    if MATL.IsNew(plyFile,plyFile_Smooth): 
        print("Smoothing and reducing Surface...")
        SmoothPly(plyFile,plyFile_Smooth)

    # Make ply with curvature information
    if MATL.IsNew(plyFile_Smooth,plyFile_Curvs):
        print("Computing Curvatures of Surface...")
        MakePLYWithCurvatureInfo(plyFile_Smooth,plyFile_Curvs,5)


    if MATL.IsNew(plyFile_Curvs,txtFile_DataPoints) and (PointsList is not None):
        print("Interpolating Points Info From Mesh...")
        head,prop = InterpolatePropertiesFromPLY(plyFile_Curvs,PointsList)

        # txtFile_DataPoints

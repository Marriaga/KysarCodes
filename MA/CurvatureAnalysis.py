#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
from pandas.plotting import table as pdtable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
#import MA.ImageProcessing as MAIP
import MA.Tools as MATL
import MA.CSV3D as MA3D
import MA.MeshIO as MAMIO
import MA.OrientationAnalysis as MAOA

sys.path.append('C:\\Program Files\\VCG\\MeshLab') # MeshLab

def checkNewFiberImages(DirectionalityBase,csvFile_PointsDirectionalityResults,PointsList):
    for f in [DirectionalityBase + "_20X_" + position + "_Fibers.png" for position in PointsList[:,0]]:
        if MATL.IsNew(f,csvFile_PointsDirectionalityResults): return True
    return False 

def loadPointDataLocations(File):
    Data = pd.read_csv(File)
    Names = np.unique(Data["Name"])
    Results = {}
    for eachName in Names:
        Results[eachName] = Data.loc[Data['Name'] == eachName]
    return Results

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
    Ang[Ang>90]-=180
    return Ang

def GetXYAng(Vec):
    X=Vec[0]
    Y=Vec[1]
    Ang = np.arctan2(Y,X) * 180/np.pi
    if Ang<0: Ang+=180
    if Ang>90: Ang-=180
    return Ang

def getSubscript(i,UseLetters=True):
    letter=['x','y','z']
    if UseLetters:
        if i<3: return letter[i]
        raise ValueError("Cannot use letters for higher than 3 dimensions")
    else:
        return str(i)

def buildIndex(root,shape=None,UseLetters=True):
    if shape is None: return [root]
    L=len(shape)
    Index=[]
    for i in range(shape[0]):
        ilbl=getSubscript(i,UseLetters=UseLetters)
        if L==1:
            Index.append(root+ilbl)
        elif L==2:
            for j in range(shape[1]):
                jlbl=getSubscript(j,UseLetters=UseLetters)
                Index.append(root+ilbl+jlbl)
        else:
            raise ValueError("Higher than 3D is not Implemented")
    return Index

def CollectArray(Data,root,shape=None,UseLetters=True):
    Index=buildIndex(root,shape=shape,UseLetters=UseLetters)
    newshape=[-1]
    if shape is not None: newshape.extend(shape)
    return Data[Index].values.reshape(newshape)


### MAIN FUNCTIONS ###


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
    
    #Normal Vector
    MyM.Nodes.AppendArray(MyM.NNorms,'n')
    #Shape Operator
    MyM.Nodes.AppendArray(MyM.NSdiheS,'Shape')
    MyM.Nodes.AppendArray(MyM.NHv,'H')

    print( "  Importing Data PLY ...")
    Ply.ImportMesh(MyM.Nodes,MyM.Elems)
    print( "  Saving Data Ply ...")
    Ply.SaveFile(plyFile_Curvs)

def InterpolatePropertiesFromPLY(plyFile_Curvs,csvFile_PointsData,PointsList):
    #Compute properties of specific points
    Ply = MAMIO.PLYIO()
    Ply.LoadFile(plyFile_Curvs)
    Nodes,Elems = Ply.ExportMesh()
    MyM = MAMIO.MyMesh(Nodes,Elems)
    Header = MyM.GetAllFieldsForInterpolation()
    Header.insert(0,"Position")

    properties=[]
    for p in PointsList:
        xy=(p[1],p[2])
        InterpValues=MyM.InterpolateFieldForPoint(xy,"All")
        InterpValues.insert(0, p[0])
        properties.append(InterpValues)

    #Save File
    Data = pd.DataFrame(data=properties, columns=Header)
    Data.to_csv(csvFile_PointsData,index=False)

def ComputeDirectionality(DirectionalityBase,csvFile_PointsDirectionalityResults,PointsDF,BaseAngleFiles=None):
    Npoints=len(PointsDF)
    RESULTS = pd.DataFrame(index=range(Npoints),
        columns=["Name","Position",
            "1VM_Weig","1VM_Disp","1VM_Ang","1VM_Weigu",
            "2VM_Weig1","2VM_Disp1","2VM_Ang1",
            "2VM_Weig2","2VM_Disp2","2VM_Ang2","2VM_Weigu"])
    BaseAuxDirFolder = os.path.join(DirectionalityBase+"_FiberAnalysis","")
    OAnalysis = MAOA.OrientationAnalysis(BaseAngFolder=BaseAngleFiles,OutputRoot=BaseAuxDirFolder,verbose=True)
    OAnalysis.SetWindowProperties(WType="Tukey",Alpha=0.2,PType="Separable")

    df1VM = pd.DataFrame(index=[0], columns=[
            "VM Weight (%)","VM Dispersion (k)","VM Angle (deg)","Uniform Weight (%)"])
    df2VM = pd.DataFrame(index=[0], columns=[
            "VM1 Weight (%)","VM1 Dispersion (k)","VM1 Angle (deg)",
            "VM2 Weight (%)","VM2 Dispersion (k)","VM2 Angle (deg)","Uniform Weight (%)"])

    for i in range(Npoints):
        row=PointsDF.iloc[i]
        name = row["Name"]
        RESULTS.iloc[i]["Name"]=name
        position = row["Position"]
        RESULTS.iloc[i]["Position"]=position

        imgfile = DirectionalityBase + "_20X_"+ position +"_Fibers.png"
        OAnalysis.SetImage(imgfile)
        BaseAuxImage= os.path.join(OAnalysis.OutputRoot+"_"+OAnalysis.GetWindowPropName())

        # Change below if you want to use Gradient instead of FFT
        Results = OAnalysis.ApplyFFT()

        fig, axes = plt.subplots(3, 2, figsize=(8 , 12))
        axes[0,0].imshow(mpimg.imread(BaseAuxImage+".png"),cmap="Greys_r",)
        axes[0,0].set_axis_off()
        axes[0,0].set_title("Windowed Fiber Image")
        axes[0,1].imshow(mpimg.imread(BaseAuxImage+"_PS.png"),cmap="Greys_r",)
        axes[0,1].set_axis_off()
        axes[0,1].set_title("Power Spectrum")


        Angles_R,Intensities = Results.GetAI()
        vmf = MAOA.Fitting(Angles_R,Intensities)
        p,k,m,u = vmf.FitVMU(1,plot=False)
        m=np.degrees(m)
        RESULTS.iloc[i][["1VM_Weig","1VM_Disp","1VM_Ang","1VM_Weigu"]]=[p[0],k[0],m[0],u]
        vmf.PlotVMU(axes[1,0])
        axes[1,0].set_title("1 VonMises + Uniform")

        df1VM.iloc[0] = np.round([p[0]*100,k[0],m[0],u*100],2)
        mytable = axes[2,0].table(cellText=df1VM.values.T, rowLabels=df1VM.columns, colWidths = [0.2], loc='right')
        mytable.auto_set_font_size(False)
        mytable.set_fontsize(10)
        axes[2,0].set_position([-0.05,0.1,0.4,0.33])
        axes[2,0].axis('off')

        p,k,m,u = vmf.FitVMU(2,plot=False)
        m=np.degrees(m)
        RESULTS.iloc[i][["2VM_Weig1","2VM_Disp1","2VM_Ang1"]]=[p[0],k[0],m[0]]
        RESULTS.iloc[i][["2VM_Weig2","2VM_Disp2","2VM_Ang2"]]=[p[1],k[1],m[1]]
        RESULTS.iloc[i]["2VM_Weigu"]=u
        vmf.PlotVMU(axes[1,1])
        axes[1,1].set_title("2 VonMises + Uniform")

        df2VM.iloc[0] = np.round([p[0]*100,k[0],m[0],p[1]*100,k[1],m[1],u*100],2)
        mytable = axes[2,1].table(cellText=df2VM.values.T, rowLabels=df2VM.columns, colWidths = [0.2], loc='right')
        mytable.auto_set_font_size(False)
        mytable.set_fontsize(10)
        axes[2,1].set_position([0.4,0.1,0.4,0.33])
        axes[2,1].axis('off')

        plt.savefig(BaseAuxImage +".pdf")

    RESULTS.to_csv(csvFile_PointsDirectionalityResults,index=False)

def ProcessInterpolatedPointsData(csvFile_PointsIntData,csvFile_PointsCurvatureResults):
    Data = pd.read_csv(csvFile_PointsIntData)
    S_Position = CollectArray(Data,'Position')
    V_Coordinates = CollectArray(Data,'',[3])
    #V_Normal = CollectArray(Data,'n',[3])
    T_Shape=CollectArray(Data,'Shape',[3,3])
    V_H = CollectArray(Data,'H',[3])

    NPoints = len(S_Position)
    outdata = []
    header = ["Position","X","Y","Angle_KMax","Angle_KMin","Angle_KMinMag1","Angle_KMinMag2","Type"]
    outdf = pd.DataFrame(index=np.arange(0, NPoints), columns=header)
    for n in range(NPoints):
        Vmax,Vmin,Vnor,kmax,kmin,Type,alph,VMM1, VMM2= MAMIO.MyMesh.IndividualCurvPrincipalDirections(T_Shape[n],V_H[n])
        Angle_KMax=GetXYAng(Vmax)
        Angle_KMin=GetXYAng(Vmin)
        Angle_KMinMag1=GetXYAng(VMM1)
        Angle_KMinMag2=GetXYAng(VMM2)
        Coord = V_Coordinates[n]
        outdf.loc[n] = [S_Position[n],Coord[0],Coord[1],Angle_KMax,Angle_KMin,Angle_KMinMag1,Angle_KMinMag2,Type]
    outdf.to_csv(csvFile_PointsCurvatureResults,index=False)


### MASTER FUNCTIONS ###

def getBaseNameFromFolder(File,Base):
    if Base is None:
        #Assumes that the Folder where the file is in is the main root name
        Folder = os.path.dirname(File)
        root = os.path.split(Folder)[-1]
        Base = os.path.join(Folder,root)
    return Base
    
def MakePLYFromTif(tiffile,Base=None,ScriptFileName=None,Force=False):
    Base = getBaseNameFromFolder(tiffile,Base)
    plyFile = Base+"_PLY0_Original.ply"
    plyFile_Smooth = Base+"_PLY1_Reduced.ply"
    plyFile_Curvs = Base+"_PLY2_Curvatures.ply"

    # Make ply from height map if heightmap is newer
    if Force or MATL.IsNew(tiffile,plyFile): 
        print("Making 3D Surface...")
        MA3D.Make3DSurfaceFromHeightMapTiff(tiffile,OFile=plyFile,NoZeros=True)

    # Smooth ply file if input ply is new
    if Force or MATL.IsNew(plyFile,plyFile_Smooth): 
        print("Smoothing and reducing Surface...")
        SmoothPly(plyFile,plyFile_Smooth,ScriptFileName=None)

    # Make ply with curvature information
    if Force or MATL.IsNew(plyFile_Smooth,plyFile_Curvs):
        print("Computing Curvatures of Surface...")
        MakePLYWithCurvatureInfo(plyFile_Smooth,plyFile_Curvs,5)

    return plyFile_Curvs

def ProcessPoints(plyFile_Curvs,PointsDF,Base=None,Force=False,BaseAngleFiles=None):
    Base = getBaseNameFromFolder(plyFile_Curvs,Base)
    csvFile_PointsIntData = Base+"_Points_Interpolated_Data.csv"
    DirectionalityBase = Base
    csvFile_PointsDirectionalityResults = Base+"_Points_DirectionalityResults.csv"
    csvFile_PointsCurvatureResults = Base+"_Points_CurvatureResults.csv"

    PointsList = PointsDF[['Position','X','Y']].values

    # Go to each point in mesh and extract interpolated results from ply mesh
    if Force or MATL.IsNew(plyFile_Curvs,csvFile_PointsIntData):
        print("Interpolating Points Info From Mesh...")
        InterpolatePropertiesFromPLY(plyFile_Curvs,csvFile_PointsIntData,PointsList)

    # Compute Directionality from Fiber Images 
    if Force or checkNewFiberImages(DirectionalityBase,csvFile_PointsDirectionalityResults,PointsList):
        print("Compute Directionality for Points Info From Mesh...")
        ComputeDirectionality(DirectionalityBase,csvFile_PointsDirectionalityResults,PointsDF,BaseAngleFiles=BaseAngleFiles)
       
    # Collect Angles from each point 
    if Force or MATL.IsNew(csvFile_PointsIntData,csvFile_PointsCurvatureResults):
        print("Process Interpolated Points Data...")
        ProcessInterpolatedPointsData(csvFile_PointsIntData,csvFile_PointsCurvatureResults)

    
def DO_EVERYTHING(FoldersNames,THEBASE,Force=False):
    '''
    FoldersNames- Folers that contain the tif file with the heights and the fiber images.
                  tif files should be named "Folder_AverageHeight_512_Smooth.tif" and fiber
                  images should be named "Folder_20X_Position_Fibers.png"
    THEBASE- The path where all the folder are. Also should contain the Points.csv file,
             which has the fields "Name,Position,X,Y", where "Name" is the folder name,
             "Position" is the label for position in membrane and "X,Y" are the coordinates
             in 10X membrane coordinates of said points.
    '''
    SmoothScriptFileName=os.path.join(THEBASE,"MashlabScript")
    MakeSmoothMeshlabScript(SmoothScriptFileName)
    Points=loadPointDataLocations(os.path.join(THEBASE,"Points.csv"))
    BaseAngleFiles = os.path.join(THEBASE,"BaseAngleFiles")

    for FOLD in FoldersNames:
        tiffile = os.path.join(THEBASE,FOLD,FOLD+"_AverageHeight_512_Smooth.tif")
        plyFile_Curvs = MakePLYFromTif(tiffile,Force=Force,ScriptFileName=SmoothScriptFileName)
        ProcessPoints(plyFile_Curvs,Points[FOLD],Force=Force,BaseAngleFiles=BaseAngleFiles)


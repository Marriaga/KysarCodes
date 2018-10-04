#!/usr/bin/python
from __future__ import absolute_import, division, print_function

import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sps
import scipy.stats as spst
import seaborn as sns
import statsmodels.formula.api as smf
from pandas.plotting import table as pdtable

import MA.CSV3D as MA3D
import MA.FigureProperties as MAFP
import MA.ImageProcessing as MAIP
import MA.MeshIO as MAMIO
import MA.OrientationAnalysis as MAOA
import MA.to_precision as tp  # From https://bitbucket.org/william_rusnack/to-precision
import MA.Tools as MATL

sys.path.append('C:\\Program Files\\VCG\\MeshLab') # MeshLab

MAFP.setupPaperStyle() # Setup Figures properties 


### Data processing functions

def GetVectorFromDF(row,veclabel):
    ''' Collect a vector from a pandas Dataframe'''
    Vx=row[veclabel+"_x"]
    Vy=row[veclabel+"_y"]
    Vz=row[veclabel+"_z"]
    return np.array([Vx,Vy,Vz])

def ProjectToPlane_N(vector,normal):
    ''' Project a vector to a plane with the given normal vector '''
    proj = vector - np.dot(normal,vector.T)*normal
    return proj/np.linalg.norm(proj)

def AngToVect_N(angle,normal):
    ''' Reverse project an angle in the horizontal plane to a plane with the given normal'''
    V=np.array([
        np.cos(np.radians(angle)),
        np.sin(np.radians(angle)),
        0])
    nz = np.array([0,0,1])
    Va = V-np.dot(V,normal)/np.dot(nz,normal)*nz
    return Va/np.linalg.norm(Va)

def getAngleBetweenVectors(V1,Vbase):
    ''' Get angle between two vectors with sign given by righthand rule w.r.t. Z'''
    dotp = np.dot(V1,Vbase)/(np.linalg.norm(V1)*np.linalg.norm(Vbase))
    if dotp >=  1.0 or dotp <= -1.0: return 0 # Angle is zero 

    #Compute sign with righthand rule w.r.t. Z
    cros = np.cross(Vbase,V1)
    sign = 1 if cros[2]>=0 else -1
    angle=sign*np.degrees(np.arccos(dotp))
    
    # Fix for -90:90 degrees
    if angle>90: angle-=180
    if angle<-90: angle+=180
    return angle

def makeAngPositive(ang):
    # Convert from -90:90 to 0:180
    if ang<0:
        return ang+180
    else:
        return ang

def TransformVectorsandAnglesFromDF(row):
    ''' Calculate Vectors and Angles after transforming based on 
        rigid body motion and by projecting to the tangent plane.'''

    # Obtain vectors from DataFrame
    Pos_raw = np.array([row["X"],row["Y"],row["Z"]])
    Vnor_raw = GetVectorFromDF(row,"VN")
    VMin_raw = GetVectorFromDF(row,"VMin")
    VMax_raw = GetVectorFromDF(row,"VMax")
    VMM1_raw = GetVectorFromDF(row,"VMM1")
    VMM2_raw = GetVectorFromDF(row,"VMM2")
    VVM1_raw = AngToVect_N(row["VM1_Ang"],Vnor_raw)
    R = GetVectorFromDF(row,"R")
    T = GetVectorFromDF(row,"T")

    # Transform vectors to new position
    RTTrans = MAIP.CoordsObj.ApplyRotationAndTranslation
    Pos = RTTrans(Pos_raw,R,T,0)
    zero = np.array([0.0,0.0,0.0])
    VMin = RTTrans(ProjectToPlane_N(VMin_raw,Vnor_raw),R,zero,0)
    VMax = RTTrans(ProjectToPlane_N(VMax_raw,Vnor_raw),R,zero,0)
    VMM1 = RTTrans(ProjectToPlane_N(VMM1_raw,Vnor_raw),R,zero,0)
    VMM2 = RTTrans(ProjectToPlane_N(VMM2_raw,Vnor_raw),R,zero,0)
    VVM1 = RTTrans(ProjectToPlane_N(VVM1_raw,Vnor_raw),R,zero,0)
    
    # Compute angles between vectors on the tangent membrane plane
    Vref = VMin if abs(row["KMin"])<abs(row["KMax"]) else VMax
    AMM1 = getAngleBetweenVectors(VMM1,Vref) # Zero curvature 1
    AMM2 = getAngleBetweenVectors(VMM2,Vref) # Zero curvature 2
    AVM1 = getAngleBetweenVectors(VVM1,Vref) # Fibers direction
    AMM = (abs(AMM1)+abs(AMM2))/2 #Zero Curvature
    AVM = abs(AVM1) # Fiber direction

    # Project Vectors to horizontal plane
    VM_flat = ProjectToPlane_N(VVM1,np.array([0,0,1]))
    PCN_flat = ProjectToPlane_N(VMin,np.array([0,0,1]))
    PCP_flat = ProjectToPlane_N(VMax,np.array([0,0,1]))
    MM1_flat = ProjectToPlane_N(VMM1,np.array([0,0,1]))
    MM2_flat = ProjectToPlane_N(VMM2,np.array([0,0,1]))

    # Compute angles wrt X
    FiberAngleFlat = makeAngPositive(getAngleBetweenVectors(VM_flat,np.array([1,0,0])))
    PCNegAngleFlat = makeAngPositive(getAngleBetweenVectors(PCN_flat,np.array([1,0,0])))
    PCPosAngleFlat = makeAngPositive(getAngleBetweenVectors(PCP_flat,np.array([1,0,0])))
    ZeroC1AngleFlat = makeAngPositive(getAngleBetweenVectors(MM1_flat,np.array([1,0,0])))
    ZeroC2AngleFlat = makeAngPositive(getAngleBetweenVectors(MM2_flat,np.array([1,0,0])))

    pos=row["Position"]
    if pos in ["Left","Center"]:
        if ZeroC1AngleFlat>75 and ZeroC1AngleFlat<130:
            ZeroC1AngleFlat,ZeroC2AngleFlat=ZeroC2AngleFlat,ZeroC1AngleFlat

    return pd.Series([AMM,AVM,Pos[0],Pos[1],Pos[2],FiberAngleFlat,PCNegAngleFlat,PCPosAngleFlat,ZeroC1AngleFlat,ZeroC2AngleFlat])

def ComputeTheta(row):
    ''' Compute the parameter Theta'''
    kmax=np.absolute(row["KMax"])
    kmin=np.absolute(row["KMin"])
    return np.degrees(np.arctan(np.sqrt(np.amin([kmax,kmin])/np.amax([kmax,kmin]))))

def ComputeCircStandDev(row):
    ''' Compute Circular Standard Deviation'''
    c = row["VM1_Conc"]
    return np.degrees(np.sqrt(-2*np.log(sps.iv(1,c)/sps.iv(0,c))))/2


def ConvertAngles(row):
    R = GetVectorFromDF(row,"R")
    return pd.Series(np.round(R*180/np.pi,2))











# NAMING
SUF_csvFile_PointsIntData = "_Points_Interpolated_Data.csv"
SUF_csvFile_PointsDirectionalityResults = "_Points_DirectionalityResults.csv"
SUF_csvFile_PointsCurvatureResults = "_Points_CurvatureResults.csv"
SUF_csvFile_PointsCombinedResults = "_Points_CombinedResults.csv"
SUF_csvFile_GeometryFitResults = "_Membrane_GeometryFitData.csv"
SUF_tiffile_unscaled = "_AverageHeight_512_Smooth.tif"
SUF_tiffile = "_AverageHeight_512_Smooth_Scaled.tif"
SUF_plyFile_Orig = "_PLY0_Original.ply"
SUF_plyFile_Smooth = "_PLY1_Smooth.ply"
SUF_plyFile_Curvs = "_PLY2_Curvatures.ply"
SUF_plyFile_RT = "_PLY3_Rotated.ply"
SUF_vtuFile_base = "_VTU"
SUF_pngFile_Streamlines = "_VTU_Streamlines.png"
SUF_pngFile_Arrows = "_VTU_Arrows.png"
SUF_pngFile_Streamlines_WithVM = "_VTU_Streamlines_WithVM.png"
SUF_pngFile_Arrows_WithVM = "_VTU_Arrows_WithVM.png"


def flipIfNeeded(inputimage,Force=False):
    root,ext=os.path.splitext(inputimage)
    flippedimage=root+"_Flipped"+ext
    if Force or MATL.IsNew(inputimage,flippedimage): MAIP.FlipImage(inputimage,flippedimage)
    return flippedimage

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

def SmoothPly(plyInput,plySmooth,ScriptFileName=None,Verbose=False,MeshLabPath=None):
    if ScriptFileName is None:
        ScriptFileName=os.path.join(os.path.dirname(plyInput),"SmoothScript")
        MakeSmoothMeshlabScript(ScriptFileName)
    MATL.RunMeshLabScript(plyInput,ScriptFileName,Output_Path=plySmooth,Verbose=Verbose,MeshLabPath=MeshLabPath)

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

def GetExtraInfo(root):
    Name = os.path.splitext(os.path.basename(root))[0]
    isRight=False
    isKasza=False

    if Name[8]=="R": isRight = True
    if Name[-1]=="K": isKasza = True
    
    return isRight,isKasza


### MAIN FUNCTIONS ###
def LoadPlyAndComputeCurvature(plyFile_Smooth,SmoothN):
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
    return MyM    

def MakeVTUsFromPLY(plyFile_RT,SmoothN,vtuFile_base,Director_Vector_List,Aux_Director_Vector):
    
    # Set-up VTU
    Vtu = MAMIO.VTUIO()
    Vtu.VecLabs.append(["Curvature",["nx","ny","nz"]])

    # Compute Curvatures
    MyM = LoadPlyAndComputeCurvature(plyFile_RT,SmoothN)

    # Select MinMag
    for i in range(2):
        if i == 0:
            MyM.ReadjustMinCurvDirections(Director_Vector_List=Director_Vector_List)
        else:
            MyM.FlipMinCurvDirections(Aux_Director_Vector=Aux_Director_Vector)
        MyM.Nodes.AppendArray(MyM.NMinMag,'n')
        print( "Importing Data VTU ...")
        Vtu.ImportMesh(MyM.Nodes,MyM.Elems)
        print( "Saving Data VTU ...")
        Vtu.SaveFile(vtuFile_base+str(i)+".vtu")

def MakePLYWithCurvatureInfo(plyFile_Smooth,plyFile_Curvs,SmoothN):   
    MyM = LoadPlyAndComputeCurvature(plyFile_Smooth,SmoothN)

    #Normal Vector
    MyM.Nodes.AppendArray(MyM.NNorms,'n')
    #Shape Operator
    MyM.Nodes.AppendArray(MyM.NSdiheS,'Shape')
    MyM.Nodes.AppendArray(MyM.NHv,'H')

    Ply = MAMIO.PLYIO()
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
    isRight,isKasza = GetExtraInfo(DirectionalityBase)
    
    Npoints=len(PointsDF)
    RESULTS = pd.DataFrame(index=range(Npoints),
        columns=["Name","Position",
            "VM1_Weig","VM1_Conc","VM1_Ang","VM1_Weigu",
            "VM2_Weig1","VM2_Conc1","VM2_Ang1",
            "VM2_Weig2","VM2_Conc2","VM2_Ang2","VM2_Weigu"])
    BaseAuxDirFolder = os.path.join(DirectionalityBase+"_FiberAnalysis","")
    OAnalysis = MAOA.OrientationAnalysis(BaseAngFolder=BaseAngleFiles,OutputRoot=BaseAuxDirFolder,verbose=True)
    OAnalysis.SetWindowProperties(WType="Tukey",Alpha=0.2,PType="Separable")

    df1VM = pd.DataFrame(index=[0], columns=[
            "VM Weight (%)","VM Concentration (k)","VM Angle (deg)","Uniform Weight (%)"])
    df2VM = pd.DataFrame(index=[0], columns=[
            "VM1 Weight (%)","VM1 Concentration (k)","VM1 Angle (deg)",
            "VM2 Weight (%)","VM2 Concentration (k)","VM2 Angle (deg)","Uniform Weight (%)"])

    for i in range(Npoints):
        row=PointsDF.iloc[i]
        name = row["Name"]
        RESULTS.iloc[i]["Name"]=name
        position = row["Position"]
        RESULTS.iloc[i]["Position"]=position

        imgfile = DirectionalityBase + "_20X_"+ position +"_Fibers.png"
        print("  - Directionality of: " + imgfile)
        if isRight: imgfile = flipIfNeeded(imgfile)

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
        p,k,m,u = vmf.FitVMU(1)
        m=np.degrees(m)
        RESULTS.iloc[i][["VM1_Weig","VM1_Conc","VM1_Ang","VM1_Weigu"]]=[p[0],k[0],m[0],u]
        vmf.PlotVMU(axes[1,0])
        axes[1,0].set_title("1 VonMises + Uniform")

        df1VM.iloc[0] = np.round([p[0]*100,k[0],m[0],u*100],2)
        mytable = axes[2,0].table(cellText=df1VM.values.T, rowLabels=df1VM.columns, colWidths = [0.2], loc='right')
        mytable.auto_set_font_size(False)
        mytable.set_fontsize(10)
        axes[2,0].set_position([-0.05,0.1,0.4,0.33])
        axes[2,0].axis('off')

        p,k,m,u = vmf.FitVMU(2)
        m=np.degrees(m)
        RESULTS.iloc[i][["VM2_Weig1","VM2_Conc1","VM2_Ang1"]]=[p[0],k[0],m[0]]
        RESULTS.iloc[i][["VM2_Weig2","VM2_Conc2","VM2_Ang2"]]=[p[1],k[1],m[1]]
        RESULTS.iloc[i]["VM2_Weigu"]=u
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
    V_Normal = CollectArray(Data,'n',[3])
    T_Shape=CollectArray(Data,'Shape',[3,3])
    V_H = CollectArray(Data,'H',[3])

    NPoints = len(S_Position)
    header = ["Position","X","Y","Z","KMax","KMin","alpha","Angle_KMax","Angle_KMin","Angle_KMinMag1","Angle_KMinMag2","Curv_Type",
    "VMax_x","VMax_y","VMax_z","VMin_x","VMin_y","VMin_z",
    "VMM1_x","VMM1_y","VMM1_z","VMM2_x","VMM2_y","VMM2_z",
    "VN_x","VN_y","VN_z"]
    outdf = pd.DataFrame(index=np.arange(0, NPoints), columns=header)
    for n in range(NPoints):
        Vmax,Vmin,Vnor,kmax,kmin,Type,alph,VMM1, VMM2= MAMIO.MyMesh.IndividualCurvPrincipalDirections(T_Shape[n],V_H[n]) # pylint: disable=unused-variable
        Angle_KMax=GetXYAng(Vmax)
        Angle_KMin=GetXYAng(Vmin)
        Angle_KMinMag1=GetXYAng(VMM1)
        Angle_KMinMag2=GetXYAng(VMM2)
        Coord = V_Coordinates[n]
        outdf.loc[n] = [S_Position[n],Coord[0],Coord[1],Coord[2],kmax,kmin,alph*180.0/np.pi,Angle_KMax,Angle_KMin,Angle_KMinMag1,Angle_KMinMag2,Type,
        Vmax[0],Vmax[1],Vmax[2],Vmin[0],Vmin[1],Vmin[2],
        VMM1[0],VMM1[1],VMM1[2],VMM2[0],VMM2[1],VMM2[2],
        V_Normal[n,0],V_Normal[n,1],V_Normal[n,2]]
    outdf.to_csv(csvFile_PointsCurvatureResults,index=False)

def CombineAndProcessMembraneData(csvFile_PointsCombinedResults,csvFile_PointsCurvatureResults,csvFile_PointsDirectionalityResults,csvFile_GeometryFitResults):
    CurvData = pd.read_csv(csvFile_PointsCurvatureResults)
    DircData = pd.read_csv(csvFile_PointsDirectionalityResults)
    NewData = CurvData.merge(DircData,on="Position")

    # Add Transformation Data
    GFitData = pd.read_csv(csvFile_GeometryFitResults)
    for col in GFitData:
        NewData[col]=GFitData[col][0]

    Name = os.path.splitext(os.path.basename(csvFile_PointsCombinedResults))[0]
    NewData["MembraneID"]=Name.split("_")[0]

    # Add Membrane Info
    isRight,isKasza = GetExtraInfo(csvFile_PointsCombinedResults)
    NewData["Side"]="RightEar" if isRight else "LeftEar"
    NewData["Microscope"]="Kasza" if isKasza else "Uptown"
    if isRight:
        def fixleftright(pos):
            if pos=="Left": return "Right"
            if pos=="Right": return "Left"
            if pos=="Bottom_Right": return "Bottom_Left"
            if pos=="Bottom_Left": return "Bottom_Right"
            return pos
        NewData["Position"] = list(map(fixleftright,NewData["Position"]))

    NewData["Theta"]=NewData.apply(ComputeTheta,axis=1)
    NewData["CircSTDV"]=NewData.apply(ComputeCircStandDev,axis=1)
    NewData[['AMM','AVM','X_T','Y_T','Z_T','AVM_flat','PCN_flat','PCP_flat','ZeroC1_flat','ZeroC2_flat']] = NewData.apply(TransformVectorsandAnglesFromDF,axis=1)
    
    NewData.to_csv(csvFile_PointsCombinedResults,index=False)

### MASTER FUNCTIONS ###

def getBaseNameFromFolder(File,Base=None):
    if Base is None:
        #Assumes that the Folder where the file is in is the main root name
        Folder = os.path.dirname(File)
        root = os.path.split(Folder)[-1]
        Base = os.path.join(Folder,root)
    return Base

def MakePLYsFromTif(Base,ScriptFileName=None,SmoothN=5,Force=False):
    tiffile_unscaled = Base + SUF_tiffile_unscaled
    tiffile =  Base + SUF_tiffile
    plyFile_Orig = Base + SUF_plyFile_Orig
    plyFile_Smooth = Base + SUF_plyFile_Smooth
    plyFile_Curvs = Base + SUF_plyFile_Curvs
    isRight,_isKasza = GetExtraInfo(Base)

    # Scale Tif height map to fix spherical distorsion
    if Force or MATL.IsNew(tiffile_unscaled,tiffile): 
        print("Scaling Tif file...")
        scalefactor = 1.335 #(nPBS/nH20)
        MAIP.ScaleTif(tiffile_unscaled,tiffile,scalefactor)

    if isRight: tiffile = flipIfNeeded(tiffile,Force=Force)

    # Make ply from height map if heightmap is newer
    if Force or MATL.IsNew(tiffile,plyFile_Orig): 
        print("Making 3D Surface...")
        MA3D.Make3DSurfaceFromHeightMapTiff(tiffile,OFile=plyFile_Orig,NoZeros=True)

    # Smooth ply file if input ply is new
    if Force or MATL.IsNew(plyFile_Orig,plyFile_Smooth): 
        print("Smoothing and reducing Surface...")
        SmoothPly(plyFile_Orig,plyFile_Smooth,ScriptFileName=None)


    # Make ply with curvature information
    if Force or MATL.IsNew(plyFile_Smooth,plyFile_Curvs):
        print("Computing Curvatures of Surface...")
        MakePLYWithCurvatureInfo(plyFile_Smooth,plyFile_Curvs,SmoothN)

def ComputeFitParameters(Base,FitObject,Force=False):
    csvFile_GeometryFitResults = Base + SUF_csvFile_GeometryFitResults
    tiffile =  Base + SUF_tiffile
    plyFile_Smooth = Base + SUF_plyFile_Smooth
    plyFile_RT = Base + SUF_plyFile_RT
    isRight,isKasza = GetExtraInfo(Base)

    if isRight: tiffile = flipIfNeeded(tiffile,Force=Force)

    # Make ply from height map if heightmap is newer
    if Force or MATL.IsNew(tiffile,plyFile_RT): 
        print("Fit To Surface...")

        R,T,AvgDZ = FitObject.FitNewImage(tiffile)
        Ply = MAMIO.PLYIO()
        Ply.LoadFile(plyFile_Smooth)
        Nodes,Elems = Ply.ExportMesh()
        Nodes.RotateAndTranslate(R,T)
        Ply.ImportMesh(Nodes,Elems)
        Ply.SaveFile(plyFile_RT)

        #Save File
        Data = pd.DataFrame(data=[[R[0],R[1],R[2],T[0],T[1],T[2],AvgDZ]], columns=["R_x","R_y","R_z","T_x","T_y","T_z","Avg_DZ"])
        Data.to_csv(csvFile_GeometryFitResults,index=False)

def ComputeStreamlines(Base,SmoothN=5,Force=False):
    plyFile_RT = Base + SUF_plyFile_RT
    vtuFile_base = Base + SUF_vtuFile_base
    pngFile_Streamlines = Base + SUF_pngFile_Streamlines
    pngFile_Arrows = Base + SUF_pngFile_Arrows
    pngFile_Streamlines_WithVM = Base + SUF_pngFile_Streamlines_WithVM
    pngFile_Arrows_WithVM = Base + SUF_pngFile_Arrows_WithVM
    csvFile_PointsCombinedResults = Base + SUF_csvFile_PointsCombinedResults

    # Make vtus from height map if heightmap is newer
    VTU1=vtuFile_base+str(0)+".vtu"
    VTU2=vtuFile_base+str(1)+".vtu"
    if Force or MATL.IsNew(plyFile_RT,VTU1): 
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
        MakeVTUsFromPLY(plyFile_RT,SmoothN,vtuFile_base,Director_Vector_List,Aux_Vec)


    if Force or MATL.IsNew(VTU1,pngFile_Streamlines): 
        import MA.ParaviewScripts
        print("Make Streamlines...")
        MA.ParaviewScripts.runscript(VTU1,VTU2,pngFile_Streamlines,Kind="Streamlines")
        print("Make Arrows...")
        MA.ParaviewScripts.runscript(VTU1,VTU2,pngFile_Arrows,Kind="Arrows")
        
        Data = pd.read_csv(csvFile_PointsCombinedResults)
        print("Mark Fiber Directions in Streamlines...")
        MAIP.MarkPointsInImage(pngFile_Streamlines,pngFile_Streamlines_WithVM,Data,radius=10,offset=8,fontSize=48)
        print("Mark Fiber Directions in Arrows...")
        MAIP.MarkPointsInImage(pngFile_Arrows,pngFile_Arrows_WithVM,Data,radius=10,offset=8,fontSize=48)

def ProcessPoints(Base,PointsDF,Force=False,BaseAngleFiles=None):
    plyFile_Curvs = Base + SUF_plyFile_Curvs
    csvFile_PointsIntData = Base + SUF_csvFile_PointsIntData
    DirectionalityBase = Base
    csvFile_PointsDirectionalityResults = Base + SUF_csvFile_PointsDirectionalityResults
    csvFile_PointsCurvatureResults = Base + SUF_csvFile_PointsCurvatureResults
    csvFile_GeometryFitResults = Base + SUF_csvFile_GeometryFitResults
    csvFile_PointsCombinedResults = Base + SUF_csvFile_PointsCombinedResults
    isRight,isKasza = GetExtraInfo(Base)

    PointsList = PointsDF[['Position','X','Y']].values

    if isRight:
        SideSize = 1.38378*1024 if isKasza else 1.24296*1024
        PointsList[:,1]*=(-1)
        PointsList[:,1]+=SideSize

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

    # Make csv with all data
    if Force or MATL.AnyIsNew([csvFile_PointsDirectionalityResults,csvFile_PointsCurvatureResults,csvFile_GeometryFitResults],csvFile_PointsCombinedResults):
        print("Combine and Generate additional results")
        CombineAndProcessMembraneData(csvFile_PointsCombinedResults,csvFile_PointsCurvatureResults,csvFile_PointsDirectionalityResults,csvFile_GeometryFitResults)

def CombineAndProcessAllData(FoldersNames,THEBASE):
    
    csvAllResults = os.path.join(THEBASE,"AllResults.csv")
    if os.path.isfile(csvAllResults): os.remove(csvAllResults)

    PlotsBase = os.path.join(THEBASE,"Plots","Plot_")
    MATL.MakeNewDir(os.path.join(THEBASE,"Plots"))

    # Construct and save all data combined
    AllDataList=[]
    for FOLD in FoldersNames:
        Base = os.path.join(THEBASE,FOLD,FOLD)
        csvFile_PointsCombinedResults = Base + SUF_csvFile_PointsCombinedResults
        AllDataList.append(pd.read_csv(csvFile_PointsCombinedResults))
    AllData = pd.concat(AllDataList,ignore_index=True)
    AllData.to_csv(csvAllResults,index=False)
    # Position, X, Y, Z, KMax, KMin, alpha, Angle_KMax, Angle_KMin, Angle_KMinMag1, Angle_KMinMag2, Curv_Type, 
    # VMax_x, VMax_y, VMax_z, VMin_x, VMin_y, VMin_z, VMM1_x, VMM1_y, VMM1_z, VMM2_x, VMM2_y, VMM2_z, VN_x, VN_y, VN_z, 
    # Name, VM1_Weig, VM1_Conc, VM1_Ang, VM1_Weigu, VM2_Weig1, VM2_Conc1, VM2_Ang1, VM2_Weig2, VM2_Conc2, VM2_Ang2, VM2_Weigu, 
    # R_x, R_y, R_z, T_x, T_y, T_z, Avg_DZ, MembraneID, Side, Microscope, Theta, CircSTDV, AMM, AVM, X_T, Y_T, Z_T,
    # AVM_flat, PCN_flat, PCP_flat, ZeroC1_flat, ZeroC2_flat

    # Make Table with geometry fitting results:
    OnlyMembranes = AllData[AllData["Position"]=="Center"].copy()
    Rcols = ["Rdx","Rdy","Rdz"]
    OnlyMembranes[Rcols] = OnlyMembranes.apply(ConvertAngles,axis=1)
    with open(PlotsBase+'Tab_GeoFitProperties.tex', 'w') as tf:
        tf.write(OnlyMembranes.to_latex(index=False,float_format='%.3f',columns=["MembraneID","Side"]+Rcols+["Avg_DZ"]))


    # Bottom_Left and Bottome_Right become Bottom!
    AllData = AllData.replace(["Bottom_Left","Bottom_Right"],value="Bottom")

    # Compute Outlier Point(s)
    AllData["Outlier"]=[p=="Bottom" and kmax<0 for p,kmax in zip(AllData["Position"],AllData["KMax"])]
    AllData=AllData[np.logical_not(AllData["Outlier"])] #Remove outlier
    
    MakePlots = ""
    
    if "1" in MakePlots:    # Plot Fiber Direction vs Theta
        MAFP.regression_figure(AllData,"Theta","AVM",
            xlabel="$\\theta$ - Relative angle of zero curvature direction (deg)",
            ylabel="Relative angle of fiber direction (deg)",
            title= "Effect of Curvature on Fiber direction.",
            savepath=PlotsBase+"Reg_ThetaVsAVM_All.pdf")

    if "2" in MakePlots:    # Plot Fiber Direction vs Theta - Selected
        AllData_noRC = AllData[[pos in ["Top","Left","Bottom"] for pos in AllData["Position"]]]
        MAFP.regression_figure(AllData_noRC,"Theta","AVM",
            xlabel="$\\theta$ - Relative angle of zero curvature direction (deg)",
            ylabel="Relative angle of fiber direction (deg)",
            title= "Effect of Curvature on Fiber direction - Selected Regions.",
            savepath=PlotsBase+"Reg_ThetaVsAVM_TLB.pdf")

    if "3" in MakePlots:    # Plot Dispersion vs Theta
        MAFP.regression_figure(AllData,"Theta","CircSTDV",line45=False,
            xlabel="$\\theta$ - Relative angle of zero curvature direction (deg)",
            ylabel="Circular Standard Deviation (deg)",
            title= "Effect of Curvature on Fiber dispersion.",
            savepath=PlotsBase+"Reg_ThetaVsCSTDEV.pdf",
            )
    
    if "4" in MakePlots:    # Box Principal Curvatures vs Region
        MAFP.box_plot_figure(AllData,"Position",["KMax","KMin"],
                column_labels=["Maximum Principal Curvature","Minimum Principal Curvature"],
                legend_title="Type of Principal Curvature",
                xlabel="Position of the point in the membrane",ylabel="Value of Curvature (1/$\\mu$m)",
                title="Principal curvatures at different locations of the membranes",
                savepath=PlotsBase+"Box_CurvatureVsPosition.pdf",
                zeroline=True)

    if "5" in MakePlots:    # Box Relative Fiber Direction vs Theta
        MAFP.box_plot_figure(AllData,"Position",["AVM","Theta"],
                column_labels=["Fiber direction","Zero curvature direction ($\\theta$)"],
                legend_title="Relative angle:",
                xlabel="Position of the point in the membrane",ylabel="Value of Angle (deg)",
                title="Effect of Curvature on Fiber direction",
                savepath=PlotsBase+"Box_RelativeAngles.pdf",
                )

    if "6" in MakePlots:    # Box Absolute Fiber Direction vs Theta
        MAFP.box_plot_figure(AllData,"Position",["AVM_flat",'ZeroC2_flat',"PCN_flat"],
                column_labels=["Fiber direction","Zero curvature direction","Principal direction"],
                legend_title="Absolute angle:",
                xlabel="Position of the point in the membrane",ylabel="Value of Angle (deg)",
                title="Absolute angles of fiber direction and curvaure",
                savepath=PlotsBase+"Box_AbsoluteAngles.pdf",
                )

        for pos in ["Bottom","Left", "Center", "Right", "Top"]:
            Subset=AllData[AllData["Position"]==pos]
            Fibers=Subset["AVM_flat"]
            ZeroC=Subset['ZeroC2_flat']
            PrinC=Subset["PCN_flat"]
            print(pos)
            print("ANOVA")
            Fstat,pval = spst.f_oneway(Fibers,ZeroC,PrinC)
            print(Fstat,pval)
            print("Fibers,ZeroC")
            Fstat,pval = spst.ttest_ind(Fibers,ZeroC)
            print(Fstat,pval)
            Fstat,pval = spst.ttest_rel(Fibers,ZeroC)
            print(Fstat,pval)
            print("Fibers,PrinC")
            Fstat,pval = spst.ttest_ind(Fibers,PrinC)
            print(Fstat,pval)
            Fstat,pval = spst.ttest_rel(Fibers,PrinC)
            print(Fstat,pval)
            print("ZeroC,PrinC")
            Fstat,pval = spst.ttest_ind(ZeroC,PrinC)
            print(Fstat,pval)
            Fstat,pval = spst.ttest_rel(ZeroC,PrinC)
            print(Fstat,pval)
            


    # fig, ax = plt.subplots(1,1,figsize=MAFP.HalfPage)
    # ax.axhline(color="k",zorder=0)
    # ax = box_plot_columns(AllData,"Position",["KMax","KMin"],"Curvature type","Curvature value",ax=ax,order=["Bottom","Left", "Center", "Right", "Top"])
    # ax.set_title("Values of the Principal Curvatures")
    # plt.show()

    
    # Plot Principal Curvatures
    # fig, ax = plt.subplots(1,1,figsize=MAFP.HalfPage)
    # ax.axhline(color="k",zorder=0)
    # ax = box_plot_columns(AllData,"Position",["AVM","Theta"],"Angle","Angle in degrees",ax=ax,order=["Bottom","Left", "Center", "Right", "Top"])
    # ax.set_title("Values of angles")
    # plt.show()


    # print(AllData[["Position","AVM_flat","PCN_flat","PCP_flat",'ZeroC1_flat','ZeroC2_flat']].sort_values("Position"))

    # Plot Angles with X
    # fig, ax = plt.subplots(1,1,figsize=MAFP.HalfPage)
    # ax.axhline(color="k",zorder=0)
    # ax = swarm_plot_columns(AllData,"Position",["AVM_flat","PCN_flat","PCP_flat",'ZeroC1_flat','ZeroC2_flat'],"Angle","Angle in degrees",ax=ax,order=["Bottom","Left", "Center", "Right", "Top"])
    # ax.set_title("Values of angles")
    # plt.show()




def DO_EVERYTHING(FoldersNames,THEBASE,Force=False,tifrefIndex=0):
    '''
    FoldersNames- List of each FOLDER that contains the tif file with the heights and the fiber images.
                  Should follow the format YYYMMDDSm (Year,Month,Day,Side,microscope). tif files should
                  be named "FOLDER_AverageHeight_512_Smooth.tif" and fiber images should be named
                  "FOLDER_20X_Position_Fibers.png".
    THEBASE- The path where all the folders are. Also should contain the Points.csv file,
             which has the fields "Name,Position,X,Y", where "Name" is the folder name,
             "Position" is the label for position in membrane and "X,Y" are the coordinates
             in 10X membrane coordinates of said points.
    Force (default=False) - Recompute Everything
    tifrefIndex (default=0) - index (in the FolderNames list) of the membrane to use as reference
    '''
    SmoothScriptFileName=os.path.join(THEBASE,"MashlabScript")
    MakeSmoothMeshlabScript(SmoothScriptFileName)
    Points=loadPointDataLocations(os.path.join(THEBASE,"PointsNew.csv"))
    BaseAngleFiles = os.path.join(THEBASE,"BaseAngleFiles")


    FitBase = os.path.join(THEBASE,FoldersNames[tifrefIndex],FoldersNames[tifrefIndex])
    tif_unsc = FitBase + SUF_tiffile_unscaled 
    tif_scld = FitBase + SUF_tiffile

    # Scale Tif height map to fix spherical distorsion
    if Force or MATL.IsNew(tif_unsc,tif_scld): 
        scalefactor = 1.335 #(nPBS/nH20)
        MAIP.ScaleTif(tif_unsc,tif_scld,scalefactor)

    FitObject = MAIP.ImageFit(tif_scld,InactiveThreshold=0.1)

    for FOLD in FoldersNames:
        Base = os.path.join(THEBASE,FOLD,FOLD)
        MakePLYsFromTif(Base,ScriptFileName=SmoothScriptFileName,SmoothN=5,Force=Force)
        ComputeFitParameters(Base,FitObject,Force=Force)
        ProcessPoints(Base,Points[FOLD],Force=Force,BaseAngleFiles=BaseAngleFiles)
        ComputeStreamlines(Base,SmoothN=5,Force=Force)

    # Combine and Process Data
    CombineAndProcessAllData(FoldersNames,THEBASE)

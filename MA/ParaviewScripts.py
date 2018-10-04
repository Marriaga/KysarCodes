from __future__ import division
import numpy as np
import sys

if sys.version_info[0] >= 3: # Python 3
    import MA.Tools as MATL
    import os,inspect

    def runstreamline(Input1,Input2,Output):
        thismodule = os.path.abspath(inspect.getfile(inspect.currentframe()))
        quotInput1= '"' + Input1 + '"'
        quotInput2= '"' + Input2 + '"'
        quotOutput= '"' + Output + '"'

        cmd = " ".join(["py -2",thismodule,quotInput1,quotInput2,quotOutput,"-Offset1 70"])
        MATL.RunProgram(cmd)

    def runscript(Input1,Input2,Output,Kind="Streamlines"):
        thismodule = os.path.abspath(inspect.getfile(inspect.currentframe()))
        quotInput1= '"' + Input1 + '"'
        quotInput2= '"' + Input2 + '"'
        quotOutput= '"' + Output + '"'

        cmd = " ".join(["py -2",thismodule,quotInput1,quotInput2,quotOutput,"-Kind",Kind,"-Offset1 70"])
        MATL.RunProgram(cmd)



else: # Python 2
    sys.path.append('C:\\Program Files\\ParaView 5.4.1-Qt5-OpenGL2-Windows-64bit\\bin\\Lib\\site-packages') # Paraview 
    sys.path.append('C:\\Program Files\\ParaView 5.4.1-Qt5-OpenGL2-Windows-64bit\\bin\\Lib') # Paraview 
    import paraview.simple as pvs #pylint: disable=E0401
    import paraview.servermanager #pylint: disable=E0401

    def AdjustCameraAndSave(renderViewIn,Output,ImageResolution=(2048,2048),CamDirVec=None,CamUpVec=None):

        # Set Defaults
        if CamDirVec is not None:
            CamDirVec = np.array(CamDirVec)*1.0
        else:
            CamDirVec = np.array([0.0,0.0,1.0])

        if CamUpVec is not None:
            CamUpVec = np.array(CamUpVec)*1.0
        else:
            CamUpVec = np.array([0.0, 1.0, 0.0])

        #  - Adjusts Camera Direction
        renderViewIn.CameraFocalPoint.SetData([0.0,0.0,0.0])
        renderViewIn.CameraPosition.SetData(CamDirVec)
        renderViewIn.CameraViewUp = CamUpVec

        renderViewIn.CameraPosition = [700, 700, 2000]
        renderViewIn.CameraFocalPoint = [700, 700, 500]
        renderViewIn.CameraParallelProjection = 1
        renderViewIn.CameraParallelScale = 700 
        #renderViewIn.ResetCamera()

        #  - Adjusts Background
        renderViewIn.Background = [1.0, 1.0, 1.0]

        #  - Save
        pvs.SaveScreenshot(Output, viewOrLayout=renderViewIn, ImageResolution = ImageResolution)


    def makeGlyph(vtuIn,RenderViewIn,scale=40.0,color=[1.0, 0.0, 0.0]): # create a new 'Glyph'
        # create a new 'Glyph'
        glyph1 = pvs.Glyph(Input=vtuIn, GlyphType='Arrow')
        glyph1.Vectors = ['POINTS', 'Curvature']
        glyph1.ScaleMode = 'vector'
        glyph1.ScaleFactor = scale
        glyph1.GlyphMode = 'All Points'
        glyph1Display = pvs.Show(glyph1, RenderViewIn)
        glyph1Display.DiffuseColor = color

    def ArrowsScript(Input1,Input2,Output,CamDirVec=None,CamUpVec=None):

        #### disable automatic camera reset on 'Show'
        pvs._DisableFirstRenderCameraReset()

        # create a new 'XML Unstructured Grid Reader'
        VTU1 = pvs.XMLUnstructuredGridReader(FileName=[Input1])
        VTU1.PointArrayStatus = ['Curvature']

        # create a new 'XML Unstructured Grid Reader'
        VTU2 = pvs.XMLUnstructuredGridReader(FileName=[Input2])
        VTU2.PointArrayStatus = ['Curvature']

        # get active view
        renderView1 = pvs.GetActiveViewOrCreate('RenderView')

        # show data in view
        VTU1Display = pvs.Show(VTU1, renderView1)
        VTU1Display.Representation = 'Surface'
        VTU1Display.Diffuse = 0.85
        VTU1Display.Ambient = 0.25

        makeGlyph(VTU1,renderView1,scale=40.0,color=[1.0, 0.0, 0.0])
        makeGlyph(VTU2,renderView1,scale=30.0,color=[0.0, 0.0, 1.0])

        # Save Screenshot
        AdjustCameraAndSave(renderView1,Output,ImageResolution=(2048,2048),CamDirVec=CamDirVec,CamUpVec=CamUpVec)

        # set active source
        pvs.SetActiveSource(None)
        pvs.SetActiveView(None)
        pvs.Disconnect()


    def makestream(vtuIn,SliceIn,RenderViewIn,color=[1.0, 0.0, 0.0]): # create a new 'Stream Tracer With Custom Source'
        streamTracerWithCustomSource1 = pvs.StreamTracerWithCustomSource(Input=vtuIn,
            SeedSource=SliceIn)
        streamTracerWithCustomSource1.Vectors = ['POINTS', 'Curvature']
        streamTracerWithCustomSource1.SurfaceStreamlines = 1
        streamTracerWithCustomSource1.IntegrationDirection = 'BOTH'
        streamTracerWithCustomSource1.IntegratorType = 'Runge-Kutta 4-5'
        streamTracerWithCustomSource1.IntegrationStepUnit = 'Cell Length'
        streamTracerWithCustomSource1.InitialStepLength = 0.5
        streamTracerWithCustomSource1.MinimumStepLength = 0.5
        streamTracerWithCustomSource1.MaximumStepLength = 2.0
        streamTracerWithCustomSource1.MaximumSteps = 2000

        # show data in view
        streamTracerWithCustomSource1Display = pvs.Show(streamTracerWithCustomSource1, RenderViewIn)
        streamTracerWithCustomSource1Display.DiffuseColor = color
        streamTracerWithCustomSource1Display.LineWidth = 3.5

    def StreamlinesScript(Input1,Input2,Output,N1=None,N2=None,Offset1=0.0,Offset2=0.0,CamDirVec=None,CamUpVec=None,Origin=None):
        ''' Creates an image with the streamlines from two VTUs.

            Inputs:
                Input1 - VTU with first direction of curvatures
                Input2 - VTU with second direction of curvatures
                N1 - List of normal of slice for VTU1
                N2 - List of normal of slice for VTU2
                Offset1 - Value of offset for slice of VTU1
                Offset2 - Value of offset for slice of VTU2
                CamDirVec - Vector for camera direction
                CamUpVec - Vector for camera up direction
                Origin - Vector with the position for the origin'''

        #### disable automatic camera reset on 'Show'
        pvs._DisableFirstRenderCameraReset()

        # create a new 'XML Unstructured Grid Reader'
        VTU1 = pvs.XMLUnstructuredGridReader(FileName=[Input1])
        VTU1.PointArrayStatus = ['Curvature']

        ## Fix data for Slices
        if N1 is None:
            N1 = [0.9,0.4,0.2]
        if N2 is None:
            # N2 = np.cross(N1,[0,0,1])
            N2 = [-0.8,0.5,0.16]
        if Origin is None:
            Origin = paraview.servermanager.Fetch(pvs.IntegrateVariables(Input=VTU1)).GetPoint(0)

        # create a new 'XML Unstructured Grid Reader'
        VTU2 = pvs.XMLUnstructuredGridReader(FileName=[Input2])
        VTU2.PointArrayStatus = ['Curvature']

        # get active view
        renderView1 = pvs.GetActiveViewOrCreate('RenderView')

        # show data in view
        VTU1Display = pvs.Show(VTU1, renderView1)
        VTU1Display.Representation = 'Surface'
        VTU1Display.Diffuse = 0.85
        VTU1Display.Ambient = 0.25
        # show data in view
        VTU2Display = pvs.Show(VTU2, renderView1)
        VTU2Display.Representation = 'Surface'
        VTU2Display.Diffuse = 0.85
        VTU2Display.Ambient = 0.25

        # create a new 'Slice'
        slice1 = pvs.Slice(Input=VTU1)
        slice1.SliceType.Origin = Origin
        slice1.SliceType.Normal = N1
        slice1.SliceType.Offset = 0.0
        slice1.SliceOffsetValues = [Offset1]

        # create a new 'Slice'
        slice2 = pvs.Slice(Input=VTU2)
        slice2.SliceType.Origin = Origin
        slice2.SliceType.Normal = N2
        slice2.SliceType.Offset = 0.0   
        slice2.SliceOffsetValues = [Offset2] 

        # make stremlines
        makestream(VTU1,slice1,renderView1,[1.0, 0.0, 0.0])
        makestream(VTU2,slice2,renderView1,[0.0, 0.0, 1.0])

        # Save Screenshot
        AdjustCameraAndSave(renderView1,Output,ImageResolution=(2048,2048),CamDirVec=CamDirVec,CamUpVec=CamUpVec)

        # set active source
        pvs.SetActiveSource(None)
        pvs.SetActiveView(None)
        pvs.Disconnect()

    def Launcher(Input1,Input2,Output,Kind,N1=None,N2=None,Offset1=0.0,Offset2=0.0,CamDirVec=None,CamUpVec=None,Origin=None):
        if Kind == "Streamlines":
            StreamlinesScript(Input1,Input2,Output,
            N1=N1,N2=N2,
            Offset1=Offset1,Offset2=Offset2,
            CamDirVec=CamDirVec,CamUpVec=CamUpVec,
            Origin=Origin)
        elif Kind == "Arrows":
            ArrowsScript(Input1,Input2,Output,
            CamDirVec=CamDirVec,CamUpVec=CamUpVec)


    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser(description='Compute Streamlines/Arrows.')
        parser.add_argument("Input1",help="VTU with first direction of curvatures")
        parser.add_argument("Input2",help="VTU with second direction of curvatures")
        parser.add_argument("Output",help="Destination for the image with streamlines")
        parser.add_argument("-Kind",default="Streamlines",help="Type of script to be run")
        parser.add_argument("-N1",help="List of normal of slice for VTU1")
        parser.add_argument("-N2",help="List of normal of slice for VTU2")
        parser.add_argument("-Offset1",type=float,default=0.0,help="Value of offset for slice of VTU1")
        parser.add_argument("-Offset2",type=float,default=0.0,help="Value of offset for slice of VTU2")
        parser.add_argument("-CamDirVec",help="Vector for camera direction")
        parser.add_argument("-CamUpVec",help="Vector for camera up direction")
        parser.add_argument("-Origin",help="Vector with the position for the origin")
        args = parser.parse_args()

        Launcher(args.Input1,args.Input2,args.Output,
            args.Kind,
            N1=args.N1,N2=args.N2,
            Offset1=args.Offset1,Offset2=args.Offset2,
            CamDirVec=args.CamDirVec,CamUpVec=args.CamUpVec,
            Origin=args.Origin)

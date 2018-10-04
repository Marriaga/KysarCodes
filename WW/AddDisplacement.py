# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:19:49 2018

@author: Wenbin
"""

from scipy.io import loadmat
import sys
sys.path.append('C:/Users/Wenbin/Documents/GitHub/KysarCodes/WW')
sys.path.append('C:/Users/Wenbin/Documents/GitHub/KysarCodes')
sys.path.append('C:\\Program Files\\VCG\\MeshLab') # MeshLab
import MA.MeshIO as MAIO
import MA.Tools as MATL
import MA.CurvatureAnalysis as MACA
import MA.CSV3D as MA3D
import MA.ImageProcessing as MAIP
import math
import os


class AddDeformation(object):
    
    def __init__(self, plyFile = None):
        if plyFile is not None:
            self.mesh = MAIO.PLYIO(plyFile)
        self.results = dict()
        self.u = []
        
    def readResults(self,resultsFile = None):
        if resultsFile is not None:    
            self.results = loadmat(resultsFile)
        else:
            print("Please input the directory to your result File.")
        self.u = self.results['u'][0][0][0]
        
    def interpolate(self, array, points):
        #scale
        '''sx, sy, sz = (960/(1416.99*8), 960/(1416.99*8), 960/(1416.99*8))
        
        points = self.mesh.Nodes.Mat.copy()
    
        points[:][1] *= sx
        points[:][2] *= sy
        points[:][3] *= sz
        
        print(points[10])'''
        
        #interpolation
        #points_ceil = points
        
        results = [];
        
        for (num, x, y, z) in points:
            
            x0 = (x) / 2 - 10
            y0 = (y) / 2 - 10
            z0 = (z) / 2 
            
            if (x0 > 480 or x0 < 0 or y0 > 480 or y0 < 0 or z0 > 128 or z0<0):
                c = 0
            else:
                x1 = math.ceil(x0)
                x2 = math.floor(x0)
                y1 = math.ceil(y0)
                y2 = math.floor(y0)
                z1 = math.floor(z0)
                z2 = math.ceil(z0)
        
                c12 = (z2 - z0) * array[x1, y2, z2] + (z0 - z1) * array[x1, y2, z1]
                c11 = (z2 - z0) * array[x1, y1, z2] + (z0 - z1) * array[x1, y1, z1]
                c22 = (z2 - z0) * array[x2, y2, z2] + (z0 - z1) * array[x2, y2, z1]
                c21 = (z2 - z0) * array[x2, y1, z2] + (z0 - z1) * array[x2, y1, z1]
        
                c2 = (x2 - x0) * c22 + (x0 - x1) * c12
                c1 = (x2 - x0) * c21 + (x0 - x1) * c11
        
                c = (y2 - y0) * c2 + (y0 - y1) * c1
            
            results.append(c)
        
        return results    
                
    def AppendDeformation(self):
        points = self.mesh.Nodes.Mat
        root = 'u'
        for i in range(3):
            fieldName = root + str(i+1)
            u_current = self.u[i]
            toAppend = self.interpolate(u_current, points)
            self.mesh.Nodes.AppendField(toAppend, fieldName)
        
    def ply2vtu(self,vtuout):
        vtu = MAIO.VTUIO()
        vtu.ImportMesh(self.mesh.Nodes, self.mesh.Elems)
        vtu.VecLabs.append(['Deformation',['u1', 'u2', 'u3']])
        vtu.SaveFile(vtuout,vtu.VecLabs)
        
    def printMesh(self):
        print(self.mesh.Nodes.Mat)
           
if __name__ == "__main__":
    print('Testing')
    #Load Tif
    #Scale Tif
    #Tif2Ply
    #SimplifyTif
    Force=False
    
    BaseFolder = r'C:\Users\Wenbin\Desktop\TestDeformation'
    
    # tiffile_unscaled = os.path.join(BaseFolder,'0kPa.tif')
    # tiffile = os.path.join(BaseFolder,'0kPa_scaled.tif')
    tiffile = os.path.join(BaseFolder,'0kPa.tif')
    plyFile_Orig = os.path.join(BaseFolder,'0kPa_scaled.ply')
    plyFile_Smooth = os.path.join(BaseFolder,'0kPa_scaled_smooth.ply')
    matlabdata = r'C:\Users\Wenbin\Desktop\FIDVC-master\Matlab Code\dm2\resultsFIDVC.mat'
    vtuout = os.path.join(BaseFolder,'0kPa_scaled_smooth.vtu')
    
        # Scale Tif height map to fix spherical distorsion
#    if Force or MATL.IsNew(tiffile_unscaled,tiffile): 
#        print("Scaling Tif file...")
#        scalefactor = 1.335 #(nPBS/nH20)
#        MAIP.ScaleTif(tiffile_unscaled,tiffile,scalefactor)

    # Make ply from height map if heightmap is newer
    if Force or MATL.IsNew(tiffile,plyFile_Orig): 
        print("Making 3D Surface...")
        MA3D.Make3DSurfaceFromHeightMapTiff(tiffile,OFile=plyFile_Orig,NoZeros=True)

    # Smooth ply file if input ply is new
    if True or MATL.IsNew(plyFile_Orig,plyFile_Smooth): 
        print("Smoothing and reducing Surface...")
        MACA.SmoothPly(plyFile_Orig,plyFile_Smooth,ScriptFileName=None,Verbose=True,MeshLabPath='C:\\Program Files\\VCG\\MeshLab')
    
    
    test = AddDeformation(plyFile_Smooth)
    test.readResults(matlabdata)
    test.AppendDeformation()
    test.ply2vtu(vtuout)
    
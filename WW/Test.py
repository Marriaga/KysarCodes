# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:44:26 2018

@author: Wenbin
"""


from scipy.io import loadmat

import numpy as np
import math

import sys
sys.path.append('C:/Users/Wenbin/Documents/GitHub/KysarCodes')
import MA.MeshIO as MeshIO
import MA.Tools as TL
import numpy as np


parameters = loadmat('C:/Users/Wenbin/Documents/GitHub/KysarCodes/WW/matlab_workspace.mat')
results = loadmat('C:/Users/Wenbin/Documents/GitHub/KysarCodes/WW/resultsFIDVC.mat')

dm = parameters['dm']
u = results['u']

mesh = MeshIO.PLYIO()

mesh.LoadFile('C:/Users/Wenbin/Documents/GitHub/KysarCodes/WW/AverageHeight.ply')

points = mesh.Nodes.Mat.copy()
test = points[:][1:3]

Data= u[0][0][0]
Xvals=Data[0]
Yvals=Data[1]
Zvals=Data[2]

max = 100;
for (num, x, y, z) in nodes:
    if z > max :
        max = z
        
print(max)
        
            
print(nodes[0])

            
            
for (num, x, y, z) in nodes:
    print(x,y,z)
    512/(381.8776*8)
        # Scale
        sx,sy,sz=((960/(1416.99*8)),960/(1416.99*8),256/(337.77417*8))
        points = self.mesh.Nodes.Mat.copy()
        points[:,0]*=sx
        points[:,1]*=sy
        points[:,2]*=sz
        
        # Interpolate
        Points_Floor = np.floor(points)
        Points_Ceil = np.ceil(points)
        alpha = Points_Ceil-points
        beta = 1-alpha.copy()
        
        X_F = Points_Floor[:,0]
        Y_F = Points_Floor[:,1]
        Z_F = Points_Floor[:,2]
        X_C = Points_Ceil[:,0]
        Y_C = Points_Ceil[:,1]
        Z_C = Points_Ceil[:,2]
        
        C12 = np.multiply(alpha , self.u[:,X_F,Y_C,Z_C]) + np.multiply(beta, self.u[:,X_F,Y_C,Z_F])
        C12 = np.multiply(alpha , self.u[:,X_F,Y_C,Z_C]) + np.multiply(beta, self.u[:,X_F,Y_C,Z_F])
        C12 = np.multiply(alpha , self.u[:,X_F,Y_C,Z_C]) + np.multiply(beta, self.u[:,X_F,Y_C,Z_F])
        C12 = np.multiply(alpha , self.u[:,X_F,Y_C,Z_C]) + np.multiply(beta, self.u[:,X_F,Y_C,Z_F])
        
        C12 = np.multiply(alpha , self.u[:,X_F,Y_C,Z_C]) + np.multiply(beta, self.u[:,X_F,Y_C,Z_F])
        C12 = np.multiply(alpha , self.u[:,X_F,Y_C,Z_C]) + np.multiply(beta, self.u[:,X_F,Y_C,Z_F])
        
        C12 = np.multiply(alpha , self.u[:,X_F,Y_C,Z_C]) + np.multiply(beta, self.u[:,X_F,Y_C,Z_F])
            
    

        
        #Append
        
        
         
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:03:19 2018

@author: Wenbin
"""


import sys
sys.path.append('C:/Users/Wenbin/Documents/GitHub/KysarCodes/WW')
sys.path.append('C:/Users/Wenbin/Documents/GitHub/KysarCodes')
import MA.CSV3D as MA3D

inFile = r'C:\Users\Wenbin\Desktop\0kPa.tif'
outFile = r'C:\Users\Wenbin\Desktop\0kPa.ply'

MA3D.Make3DSurfaceFromHeightMapTiff(inFile,OFile=outFile,NoZeros=True)


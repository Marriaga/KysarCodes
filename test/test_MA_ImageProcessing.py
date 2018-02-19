from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
import numpy as np
import os
import sys
sys.path.append("C:\\Users\\Miguel\\Dropbox\\PostDoc\\KysarCodes")
import MA.ImageProcessing as MAIP
import MA.Tools as MATL
 
class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("test_MA_ImageProcessing")
        cls.cosimg = np.round(MAIP.MakeCosImage((5,5),ang=30,freq=2),decimals=4)
        cls.temppath = os.path.join("test","temp")
        #MATL.MakeNewDir(cls.temppath)
        cls.testimgpath = os.path.join(cls.temppath,"MyTestImage.png")
        cls.testimgraw = os.path.join(cls.temppath,"MyTestImage.raw")
        
        cls.compareimagematrix = np.array(
        [[ 0.7517,  0.9887,  0.5503,  0.0424,  0.1669],
        [ 0.7118,  0.1347,  0.0624,  0.5949,  0.9962],
        [ 0.0071,  0.4274,  0.948 ,  0.8495,  0.268 ],
        [ 0.8495,  0.948 ,  0.4274,  0.0071,  0.268 ],
        [ 0.5949,  0.0624,  0.1347,  0.7118,  0.9962]]) 

    def test_Cos_Generator(self):
        self.assertTrue(np.all(self.cosimg==self.compareimagematrix))
        
    def test_Save_And_Load_Png_File(self):
        MAIP.SaveImage(self.cosimg,self.testimgpath,resc=True)
        LoadedImage=MAIP.GetImageMatrix(self.testimgpath,Silent=True)
        self.assertTrue(np.all(LoadedImage==MAIP.Rescale8bit(self.compareimagematrix)))
        
    def test_ConvertToRGB(self):
        rgbimg = MAIP.ConvertToRGB(self.cosimg)
        MAIP.SaveImageRGB(rgbimg,self.testimgpath)
        rgbimgnew = MAIP.GetRGBAImageMatrix(self.testimgpath, Silent = True)
        self.assertTrue(np.all(rgbimg==rgbimgnew))
    
    def test_OpenPILRaw(self):
        bindata=b'C\x7f\x00\x00A\xc0\x00\x00\x00\x00\x00\x00CR\x00\x00B\x92\x00\x00\x00\x00\x00\x00CC\x00\x00C=\x00\x00\x00\x00\x00\x00'
        with open(self.testimgraw,'wb') as f:
            f.write(bindata)
        mypix=MAIP.GetImageMatrix(self.testimgraw,Silent=True,ConvertL=True)
        self.assertTrue(np.all([[255,  24,   0], [210,  73,   0], [195, 189,   0]]==mypix))
        
    def test_ImageFit(self):
        # Basic Image
        RImg=MAIP.MakeRImage((50,50))
        CO_R = MAIP.CoordsObj(Img=RImg,ZScaling=50,InactiveThreshold=0)

        # Reference Image (with slight tilt for initialization)
        AdjustedCoords = CO_R.getTransformedCoords([0,np.radians(2),0],[0,0,7])
        # Rotated and Translated Coordinates
        NewCoords = CO_R.getTransformedCoords([0,np.radians(3),np.radians(90)],[3.2,1.1,2.0])

        # Adjust the original Coordinates to include small tilt
        CO_R.setFromCoords(AdjustedCoords)
        RMat = CO_R.getMat()

        # Create fitting object with reference image
        FitObj = MAIP.ImageFit(RMat,InactiveThreshold=0)
        # Try to fit NewCoords to Reference Image
        R,Tr = FitObj.FitNewCoords(NewCoords,silent=True)

        # Check if correction angle is ~= -90 degrees
        self.assertAlmostEqual(np.degrees(R[2]),-90.0,1)

        # Check if cost function is below 2.5
        self.assertTrue(FitObj.CostFunction(np.concatenate((R,Tr)))<2.5)

        # Check if the SumSquares of the diference between certain points is almost zero
        mylist=np.array([[8,18],[23,18],[15,33]]).astype(np.int32)
        NewMat=FitObj.CImage.getTransformedMat(R,Tr)
        ResObtained=NewMat[mylist[:,0],mylist[:,1]]
        ResExpected=CO_R.getMat()[mylist[:,0],mylist[:,1]]
        SumSquares=np.sum((ResObtained-ResExpected)**2)
        self.assertAlmostEqual(SumSquares,0.0,1)
        
        
    @classmethod
    def tearDownClass(cls):
        pass
        #if os.path.isdir(cls.temppath): MATL.DeleteFolderTree(cls.temppath)

 
if __name__ == '__main__':
    unittest.main()
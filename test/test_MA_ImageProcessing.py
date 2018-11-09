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
        print("\n === test_MA_ImageProcessing === ")
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
        # MATL.MakeNewDir("Test")
        # ReferenceImage = "Test\\ReferenceR.tif"
        # FittingImage = "Test\\RotatedR.tif"

        # print("Base R Image...")
        Nsize=30
        RImg=MAIP.MakeRImage((Nsize,Nsize)) # Make R shaped image matrix in Image Format
        CO_R = MAIP.CoordsObj(Img=RImg,ZScaling=50,InactiveThreshold=0.0)

        # print("Make adjusted Base Coordinates...")
        BaseTranslation = np.array([0,0,7])
        BaseRotation = np.array([np.radians(2),0,0])
        BaseCoords = CO_R.getTransformedCoords(BaseRotation,BaseTranslation,pavg=0)
        CO_Rbase = MAIP.CoordsObj(Img=RImg,ZScaling=50,InactiveThreshold=0.0)
        CO_Rbase.setFromCoords(BaseCoords)
        #CO_Rbase.saveAsImage(ReferenceImage)
        RMat = CO_Rbase.getMat()

        # print("Make New Fitting Coordinates...")
        NewTranslation = np.array([Nsize+0.4,13.7,30.0])
        NewRotation = np.array([np.radians(1),np.radians(10),np.radians(90)])
        NewCoords = CO_R.getTransformedCoords(NewRotation+BaseRotation,NewTranslation+BaseTranslation,pavg=0)
        CO_RNew = MAIP.CoordsObj(Img=RImg,ZScaling=50,InactiveThreshold=0.0)
        CO_RNew.setFromCoords(NewCoords)
        # CO_RNew.saveAsImage(FittingImage)

        # print("MAKE FIT...")
        FitObj = MAIP.ImageFit(MAIP.np2Image(RMat),InactiveThreshold=0)
        # FitObj = MAIP.ImageFit(ReferenceImage,InactiveThreshold=0)
        R,T,DZavg = FitObj.FitNewCoords(NewCoords,silent=True)
        # R,T,DZavg = FitObj.FitNewImage(FittingImage,silent=True)

        Rd=np.degrees(R)
        RMR = MAIP.CoordsObj.getRotationMatrix(NewRotation)
        Tf = -RMR.T@NewTranslation
        Rfd= np.degrees(MAIP.CoordsObj.rotationMatrixToEulerAngles(RMR.T))

        for i in range(3): self.assertAlmostEqual(Rd[i]/90.0,Rfd[i]/90.0,2)
        for i in range(3): self.assertAlmostEqual(T[i]/30.0 ,Tf[i]/30.0 ,1)
       
        
    @classmethod
    def tearDownClass(cls):
        pass
        #if os.path.isdir(cls.temppath): MATL.DeleteFolderTree(cls.temppath)

 
if __name__ == '__main__':
    unittest.main()
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
import numpy as np
import os
import MA.ImageProcessing as MAIP
import MA.Tools as MATL
 
class T(unittest.TestCase):
    def setUp(self):
        self.cosimg = np.round(MAIP.MakeCosImage((5,5),ang=30,freq=2),decimals=4)
        self.temppath = os.path.join("test","temp")
        MATL.MakeNewDir(self.temppath)
        self.testimgpath = os.path.join(self.temppath,"MyTestImage.png")
        
        self.compareimagematrix = np.array(
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
        
    def tearDown(self):
        if os.path.isdir(self.temppath): MATL.DeleteFolderTree(self.temppath)

 
if __name__ == '__main__':
    unittest.main()
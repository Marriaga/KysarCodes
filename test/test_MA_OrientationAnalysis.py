from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
import numpy as np
import os
import MA.OrientationAnalysis as MAOA
import MA.ImageProcessing as MAIP
import MA.Tools as MATL
 
class T(unittest.TestCase):
    def setUp(self):
        self.cosimg = np.round(MAIP.MakeCosImage((5,5),ang=30,freq=2),decimals=4)
        self.temppath = os.path.join("test","temp")
        MATL.MakeNewDir(self.temppath)
        self.rootpath = os.path.join(self.temppath,"Orientation")
        
        self.OAnalysis = MAOA.OrientationAnalysis(
            BaseAngFolder=self.temppath,
            OutputRoot=self.rootpath,
            verbose=False,
            )
        self.OAnalysis.SetImage(self.cosimg)


    def test_FFT_Of_Cosimg(self):
        Results = self.OAnalysis.ApplyFFT(PSCenter=2,Backup=False)
        self.assertTrue(np.round(np.mean(Results.Y),decimals=7)==0.1154337)
        
    def test_Gradient_Of_Cosimg(self):
        Results = self.OAnalysis.ApplyGradient()
        self.assertTrue(np.round(np.mean(Results.Y),decimals=7)==0.1104972)
        
    def tearDown(self):
        if os.path.isdir(self.temppath): MATL.DeleteFolderTree(self.temppath)

 
if __name__ == '__main__':
    unittest.main()
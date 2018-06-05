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
    @classmethod
    def setUpClass(cls):
        print("\n === test_MA_OrientationAnalysis === ")
        cls.cosimg = np.round(MAIP.MakeCosImage((10,10),ang=30,freq=2),decimals=4)
        cls.temppath = os.path.join("test","temp")
        #MATL.MakeNewDir(cls.temppath)
        cls.rootpath = os.path.join(cls.temppath,"Orientation")
        
        cls.OAnalysis = MAOA.OrientationAnalysis(
            BaseAngFolder=cls.temppath,
            OutputRoot=cls.rootpath,
            verbose=False,
            )
        cls.OAnalysis.SetImage(cls.cosimg)


    def test_FFT_Of_Cosimg(self):
        Results = self.OAnalysis.ApplyFFT(PSCenter=2,Backup=False)
        self.assertEqual(np.round(np.mean(Results.Y),decimals=7),0.1841117)
        
    def test_Gradient_Of_Cosimg(self):
        Results = self.OAnalysis.ApplyGradient()
        self.assertEqual(np.round(np.mean(Results.Y),decimals=7),0.086326)

    def test_Fitting(self):
        Results = self.OAnalysis.ApplyGradient()
        Angles_R,Intensities = Results.GetAI()
        vmf = MAOA.Fitting(Angles_R,Intensities)
        p,k,m,u = vmf.FitVMU(1)

        self.assertEqual(np.round(np.sum(p)+u,decimals=5),1.0)
        self.assertEqual(np.round(k[0],decimals=5),13.29075)
        
    @classmethod
    def tearDownClass(cls):
        pass
        #if os.path.isdir(cls.temppath): MATL.DeleteFolderTree(cls.temppath)

 
if __name__ == '__main__':
    unittest.main()
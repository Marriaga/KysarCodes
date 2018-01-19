from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
import numpy as np
import os

import MA.PolarizedAngleAnalysis as MAPA
import MA.Tools as MATL
 
class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temppath = os.path.join("test","temp","")
        MATL.MakeNewDir(cls.temppath)
  
    def test_ScanPoints(self):
        #Make two images and 2 xml
        #Apply A,I,refimg = ScanPoints(InpupFolder,ImageExtension,ImagePrefix=ImagePrefix,PointsList=PointsList)
        #Check if A and I are OK
        self.assertTrue(False)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.temppath): MATL.DeleteFolderTree(cls.temppath)


if __name__ == '__main__':
    unittest.main()
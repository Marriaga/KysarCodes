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
        print("\n === test_MA_PolarizedAngleAnalysis === ")
        cls.temppath = os.path.join("test","temp","")
        #MATL.MakeNewDir(cls.temppath)
  
    def test_ScanPoints(self):
        BaseName=os.path.join(self.temppath,"TPol")
        with open(BaseName+"1.xml",'w+') as fp: fp.write("<iict_transform class=\"mpicbg.trakem2.transform.AffineModel2D\" data=\"1.0 0.0 0.0 1.0 0.0 0.0\" />")
        with open(BaseName+"2.xml",'w+') as fp: fp.write("<ictl><iict_transform class=\"mpicbg.trakem2.transform.RigidModel2D\" data=\"-0.75 -1 1\" /></ictl>")
        with open(BaseName+"1.png",'wb') as fp: fp.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x00\x00\x00\x00W\xddR\xf8\x00\x00\x00\x0eIDATx\xdac\xe8;\xc6p\xec1\x00\x08S\x02\xfe8^\xa9\x81\x00\x00\x00\x00IEND\xaeB`\x82')
        with open(BaseName+"2.png",'wb') as fp: fp.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x00\x00\x00\x00W\xddR\xf8\x00\x00\x00\x0eIDATx\xdac8\xd6\xc7\xf0\xf8\x18\x00\x08\xa8\x02\xfe\xbb\xfe&p\x00\x00\x00\x00IEND\xaeB`\x82')
        
        A,I,refimg = MAPA.ScanPoints(self.temppath,"png",ImagePrefix="TPol",PointsList=[(0,0)])
        self.assertTrue(np.round(A[1],3)==-42.972)
        self.assertTrue(np.round(I[0][1],3)==198)

    @classmethod
    def tearDownClass(cls):
        pass
        #if os.path.isdir(cls.temppath): MATL.DeleteFolderTree(cls.temppath)


if __name__ == '__main__':
    unittest.main()
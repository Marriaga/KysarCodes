from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
import numpy as np
import os

import MA.CSV3D as MA3D
import MA.Tools as MATL
 
class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temppath = os.path.join("test","temp","")
        MATL.MakeNewDir(cls.temppath)
  
    def test_MakeCSVMesh(self):
        BaseName=os.path.join(self.temppath,"csv")
        with open(BaseName+".csv",'w+') as fp: fp.write("1,0\n3,1")
        with open(BaseName+".png",'wb') as fp: fp.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x00\x01sRGB\x00\xae\xce\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x00\tpHYs\x00\x00\x0e\xc3\x00\x00\x0e\xc3\x01\xc7o\xa8d\x00\x00\x00\x16IDAT\x18Wcx+\xa3\xa2\xb4\xd1\x87\x81a\xd1\x8b\xff\x9f\x18\x00$\x8b\x05\xc8\x8e2|\xd0\x00\x00\x00\x00IEND\xaeB`\x82')
        
        MA3D.MakeCSVMesh(BaseName+".csv",ColFile=BaseName+".png",Expfile=BaseName+".ply",verbose=False)
        with open(BaseName+".ply","rb") as fp: A=fp.read()
        self.assertTrue(A==b'ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\xed\x1c$\xff\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00"\xb1L\xff\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00@@\x00\xa2\xe8\xff\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x80?\xff\xf2\x00\xff\x03\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x03\x03\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00')
 
    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.temppath): MATL.DeleteFolderTree(cls.temppath)


if __name__ == '__main__':
    unittest.main()
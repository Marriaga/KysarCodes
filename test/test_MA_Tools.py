from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
import numpy as np
import os

import MA.Tools as MATL
 
class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("test_MA_Tools")
        cls.temppath = os.path.join("test","temp","")
        #MATL.MakeNewDir(cls.temppath)
        cls.myfile_old = MATL.MakeRoot(cls.temppath,"old")
        open(cls.myfile_old,'w+').close()
        # self.testimgpath = os.path.join(self.temppath,"MyTestImage.png")
   
    def test_MakeRoot(self):
        mynewfile = MATL.MakeRoot(self.temppath,"TEST")
        open(mynewfile,'w+').close()
        self.assertTrue(os.path.isfile(os.path.join(self.temppath,"TEST")))
        
    def test_getFiles(self):
        FileNameList=['firsttest','secondtest','thirdtest']
        for fn in FileNameList:
            fileloc = os.path.join(self.temppath,fn+".myext")
            open(fileloc,'w+').close()
        self.assertTrue([os.path.splitext(os.path.basename(x))[0] for x in MATL.getFiles(self.temppath,"myext")]==FileNameList)

    def test_IsNew(self):
        myfile_new = MATL.MakeRoot(self.temppath,"new")
        open(myfile_new,'w+').close()
        self.assertFalse(MATL.IsNew(self.myfile_old,myfile_new))
        self.assertTrue(MATL.IsNew(myfile_new,self.myfile_old))

    def test_TicToc(self):
        start=MATL.Tic()
        delta=MATL.Toc(start)
        self.assertTrue(delta>=0)

    def test_Timeme(self):
        avgtime=MATL.Timeme(max,[4,5,6],NN=2,NNN=2,show=False)
        self.assertTrue(avgtime>=0)

    def test_getMatfromCSV(self):
        csvdata="1,2  ,3\n4 , 5,6\n    7,8,9"
        csvfile=os.path.join(self.temppath,"mycsvfile.csv")
        with open(csvfile,'w') as f:
            f.write(csvdata)
        self.assertTrue(np.all(MATL.getMatfromCSV(csvfile)==[[1,2,3],[4,5,6],[7,8,9]]))

    def test_macauley(self):
        self.assertTrue(np.all(MATL.macauley(np.array([3,-3]),True)==np.array([3,0])))
        self.assertTrue(np.all(MATL.macauley(np.array([3,-3]),False)==np.array([0,-3])))

    def test_RunProgram(self):
        MATL.RunProgram("python -c \"print('hello')\"",False)

    def test_eig33s(self):
        mat=np.array([[1,7,6],[7,2,5],[6,5,3]])
        normdiff=np.linalg.norm(
            np.sort(list(MATL.eig33s(mat[0,0],mat[1,1],mat[2,2],mat[0,1],mat[0,2],mat[1,2])))-
            np.sort(np.linalg.eigvals(mat)))
        self.assertTrue(normdiff<1E-12)

    @classmethod
    def tearDownClass(cls):
        pass
        #if os.path.isdir(cls.temppath): MATL.DeleteFolderTree(cls.temppath)


if __name__ == '__main__':
    unittest.main()
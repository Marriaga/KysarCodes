import unittest
import shutil
import sys
import os


if __name__ == '__main__':    

    # Folder to store all temporary files
    TempFolder=os.path.join("test","temp","")

    # Delete all Files and folders in test/temp
    if os.path.exists(TempFolder): shutil.rmtree(TempFolder)
    # Make new temporary folder
    os.makedirs(TempFolder)

    #Run Tests
    test_suite = unittest.defaultTestLoader.discover('.')
    unittest.TextTestRunner(verbosity=2).run(test_suite)

    print("Tests Finished... Starting cleanup")
    
    #Delete the .pyc files in test directory
    dir_name = "test"
    for fname in os.listdir(dir_name):
        if fname.endswith(".pyc"):
            os.remove(os.path.join(dir_name, fname))
            
    #Delete all Files and folders in test/temp
    shutil.rmtree(TempFolder)

            
import unittest
import sys
import os


if __name__ == '__main__':
    test_suite = unittest.defaultTestLoader.discover('.')
    unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    #Delete the .pyc files in test directory
    dir_name = "test"
    for fname in os.listdir(dir_name):
        if fname.endswith(".pyc"):
            os.remove(os.path.join(dir_name, fname))
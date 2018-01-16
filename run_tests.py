import unittest

if __name__ == '__main__':

    test_suite = unittest.defaultTestLoader.discover('.')
    unittest.TextTestRunner(verbosity=2).run(test_suite)
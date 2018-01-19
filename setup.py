#!/usr/bin/env python          
from setuptools import setup, find_packages

setup(version='0.1',
      description='Library developed at the Kysar Lab',
      author='Miguel Arriaga',
      author_email='mta2122@columbia.edu',
      packages=find_packages(),
      install_requires=['matplotlib==2.0.2',
        'joblib>=0.11',
        'numpy>=1.13.1',
        'Pillow>=5.0.0',
        'progressbar2>=3.34.3'],
      #test_suite="tests",                          
)
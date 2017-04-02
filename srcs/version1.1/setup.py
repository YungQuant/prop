from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
      ext_modules = cythonize("cip1.1_cythonTest.py"),
      include_dirs = [np.get_include()]
      )
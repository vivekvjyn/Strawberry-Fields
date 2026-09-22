from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize([
        "strawberryfields/dtw.pyx",
        "strawberryfields/pyin.pyx",
        "strawberryfields/salience.pyx",
    ]),
    include_dirs=[np.get_include()],
)

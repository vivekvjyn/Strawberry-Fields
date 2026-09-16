from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize([
        "strawberryfields/dtw_core.pyx",
        "strawberryfields/yin_core.pyx",
        "strawberryfields/pyin_core.pyx",
    ]),
    include_dirs=[np.get_include()],
)

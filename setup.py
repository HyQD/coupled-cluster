from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np
import os
import glob


base_path = ["coupled_cluster"]

source_files = [*glob.glob(os.path.join(*base_path, "*.pyx"))]

include_dirs = [np.get_include()]

extensions = [
    Extension(
        name="coupled_cluster.cc_helper",
        sources=source_files,
        language="c",
        include_dirs=include_dirs,
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="Coupled cluster",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)

from setuptools import setup, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        'dpm_srm',
        ['wrappers/wrapper.cpp'],
        include_dirs=["./include", pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='dpm_srm',
    version='0.1.6',
    author='Digital Porous Media',
    author_email='bcchang@utexas.edu',
    description='Statistical Region Merging Segmentation',
    ext_modules=ext_modules,
    zip_safe=False,
    packages=find_packages(exclude=["include"])
)

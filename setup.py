from setuptools import setup, Extension, find_packages
import pybind11
from pybind11.setup_helpers import Pybind11Extension
import os

# include_path = os.path.abspath('include')

# Define your extension module
ext_modules = [
    Pybind11Extension(
        'dpm_srm',  # Name of the module
        ['wrappers/wrapper.cpp'],
        include_dirs=["./include", pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='dpm_srm',  # Replace with your module name
    version='0.1.0',  # Version of your package
    author='Digital Porous Media',  # Your name
    author_email='bcchang@utexas.edu',  # Your email
    description='Statistical Region Merging Segmentation',  # Description of your package
    ext_modules=ext_modules,
    zip_safe=False,
    packages=find_packages(exclude=["include"])
    # classifiers=[
    #     'License :: OSI Approved :: BSD License',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: C++',
    # ],

)

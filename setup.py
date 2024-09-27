from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "dpm_srm",
        [src.]
    )
]

# from setuptools import setup, Extension
# import pybind11

# ext_modules = [
#     Extension(
#         'your_module_name',
#         ['path/to/your_wrapper.cpp', 'path/to/SRM.cpp', 'path/to/SRM3D.cpp'],
#         include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],
#         language='c++'
#     ),
# ]

# setup(
#     name='your_module_name',
#     ext_modules=ext_modules,
#     zip_safe=False,
# )

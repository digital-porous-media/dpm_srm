cmake_minimum_required(VERSION 3.18)  # Adjust as needed
project(dpm_srm VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

# Find Python (adjust the version as needed)
find_package(Python3 REQUIRED)
# COMPONENTS Interpreter Development)

find_package(PythonLibs REQUIRED)
# If you are using pybind11
find_package(pybind11 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
# Add your source files and set up your target
# add_library(dpm SHARED wrappers/wrapper.cpp)
# target_link_libraries(dpm PRIVATE pybind11::module)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dpm_srm)
# Create the Python module
pybind11_add_module(_dpm_srm wrappers/wrapper.cpp)

set_target_properties(_dpm_srm PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

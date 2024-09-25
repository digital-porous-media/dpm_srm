cmake_minimum_required(VERSION 3.12)  # Adjust as needed
project(SRM3DModule)

# Set the policy to avoid the warning
cmake_policy(SET CMP0148 NEW)

# Find Python (adjust the version as needed)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(PythonLibs REQUIRED)
# If you are using pybind11
find_package(pybind11 REQUIRED)

# Add your source files and set up your target
add_library(MyLibrary SHARED wrapper.cpp)
target_link_libraries(MyLibrary PRIVATE pybind11::module)
set(CMAKE_CXX_STANDARD 14)

# Create the Python module
pybind11_add_module(srm3d wrapper.cpp SRM3D.cpp)

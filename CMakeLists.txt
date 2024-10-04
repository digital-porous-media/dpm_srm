cmake_minimum_required(VERSION 3.18)  # Adjust as needed
project(dpm_srm)

# Set the policy to avoid the warning
# cmake_policy(SET CMP0148 NEW)
set(CMAKE_CXX_STANDARD 20)

# Find Python (adjust the version as needed)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(PythonLibs REQUIRED)
# If you are using pybind11
find_package(pybind11 REQUIRED)

include_directories(include/)
# Add your source files and set up your target
add_library(dpmsrm SHARED dpm_srm/wrapper.cpp)
target_link_libraries(dpmsrm PRIVATE pybind11::module)


# Create the Python module
pybind11_add_module(dpm_srm dpm_srm/wrapper.cpp)

install(TARGETS dpm_srm DESTINATION .)

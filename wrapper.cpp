#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "SRM.hpp"
#include "SRM3D.hpp"
#include "SRM2D.hpp"

namespace py = pybind11;

// Template function to help wrap SRM3D with different datatypes
template <typename T>
void wrap_srm3d(py::module &m, const std::string &suffix)
{
    std::string class_name = "SRM3D_" + suffix;
    py::class_<SRM3D<T>>(m, class_name.c_str())
        .def(py::init<const py::array_t<T> &, double>(),
             py::arg("image"), py::arg("Q"))
        .def("segment", &SRM3D<T>::segment)
        .def("get_result", &SRM3D<T>::getSegmentation);
}

template <typename T>
void wrap_srm2d(py::module &m, const std::string &suffix)
{
    std::string class_name = "SRM2D_" + suffix;
    py::class_<SRM2D<T>>(m, class_name.c_str())
        .def(py::init<const py::array_t<T> &, double>(),
             py::arg("image"), py::arg("Q"))
        .def("segment", &SRM2D<T>::segment)
        .def("get_result", &SRM2D<T>::getSegmentation);
}

PYBIND11_MODULE(dpm_srm, m)
{
    m.doc() = "SRM Segmentation module";
    wrap_srm3d<uint8_t>(m, "u8");
    wrap_srm3d<uint16_t>(m, "u16");
    wrap_srm3d<int8_t>(m, "i8");
    wrap_srm3d<int16_t>(m, "i16");
    wrap_srm2d<uint8_t>(m, "u8");
    wrap_srm2d<uint16_t>(m, "u16");
    wrap_srm2d<int8_t>(m, "i8");
    wrap_srm2d<int16_t>(m, "i16");
}

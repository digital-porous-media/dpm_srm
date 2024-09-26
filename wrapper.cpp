#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "SRM.hpp"

namespace py = pybind11;

PYBIND11_MODULE(srm3d, m)
{
    m.doc() = "SRM3D segmentation module";

    py::class_<SRM3D>(m, "SRM3D")
        .def(py::init<const py::array_t<uint16_t> &, float>(),
             py::arg("image"), py::arg("Q"))
        .def("segment", &SRM3D::segment)
        .def("get_result", &SRM3D::getSegmentation);
}

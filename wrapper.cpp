#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "SRM.hpp"

namespace py = pybind11;

// std::shared_ptr<void> srm3d(py::object img, double q)
// {
//     if (py::isinstance<uint16_t>(img))
//     {
//         return std::make_shared<SRM3D<uint16_t>>(img.cast<uint16_t>(), q);
//     }
//     else if (py::isinstance<uint8_t>(img))
//     {
//         return std::make_shared<SRM3D<uint8_t>>(img.cast<uint8_t>(), q);
//     }
//     else if (py::isinstance<int16_t>(img))
//     {
//         return std::make_shared<SRM3D<int16_t>>(img.cast<int16_t>(), q);
//     }
//     else if (py::isinstance<int8_t>(img))
//     {
//         return std::make_shared<SRM3D<int8_t>>(img.cast<int8_t>(), q);
//     }
//     throw std::runtime_error("Unsupported type");
// }

PYBIND11_MODULE(srm3d, m)
{
    m.doc() = "SRM3D segmentation module";

    py::class_<SRM3D<uint8_t>>(m, "SRM3D_u1")
        .def(py::init<const py::array_t<uint8_t> &, double>(),
             py::arg("image"), py::arg("Q"))
        .def("segment", &SRM3D<uint8_t>::segment)
        .def("get_result", &SRM3D<uint8_t>::getSegmentation);

    py::class_<SRM3D<uint16_t>>(m, "SRM3D_u2")
        .def(py::init<const py::array_t<uint16_t> &, double>(),
             py::arg("image"), py::arg("Q"))
        .def("segment", &SRM3D<uint16_t>::segment)
        .def("get_result", &SRM3D<uint16_t>::getSegmentation);

    // py::class_<SRM3D<int8_t>>(m, "SRM3D_i1")
    //     .def(py::init<const py::array_t<int8_t> &, float>(),
    //          py::arg("image"), py::arg("Q"))
    //     .def("segment", &SRM3D<int8_t>::segment)
    //     .def("get_result", &SRM3D<int8_t>::getSegmentation);

    // py::class_<SRM3D<int16_t>>(m, "SRM3D_i2")
    //     .def(py::init<const py::array_t<int16_t> &, float>(),
    //          py::arg("image"), py::arg("Q"))
    //     .def("segment", &SRM3D<int16_t>::segment)
    //     .def("get_result", &SRM3D<int16_t>::getSegmentation);
}

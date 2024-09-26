#include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "SRM.hpp"

namespace py = pybind11;

// // Convert a NumPy array to a vector
// std::vector<std::vector<std::vector<int>>> numpy_to_vector(py::array_t<int> input_array)
// {
//     py::buffer_info buf_info = input_array.request();
//     int depth = buf_info.shape[0];
//     int height = buf_info.shape[1];
//     int width = buf_info.shape[2];

//     std::vector<std::vector<std::vector<int>>> result(depth, std::vector<std::vector<int>>(height, std::vector<int>(width)));

//     for (int i = 0; i < depth; i++)
//     {
//         for (int j = 0; j < height; j++)
//         {
//             for (int k = 0; k < width; k++)
//             {
//                 result[i][j][k] = static_cast<int *>(buf_info.ptr)[i * height * width + j * width + k];
//             }
//         }
//     }

//     return result;
// }

PYBIND11_MODULE(srm3d, m)
{
    m.doc() = "SRM3D segmentation module";

    py::class_<SRM3D>(m, "SRM3D")
        .def(py::init<const std::vector<std::vector<std::vector<int>>> &, double>(),
             py::arg("image"), py::arg("Q"))
        .def("segment", &SRM3D::segment)
        .def("get_result", &SRM3D::getSegmentation);

    // // Overload the constructor to accept a NumPy array
    // m.def("create_srm3d", [](py::array_t<int> input_array, double Q)
    //       {
    //     auto vector_image = numpy_to_vector(input_array);
    //     return new SRM3D(vector_image, Q); }, py::return_value_policy::take_ownership, py::arg("image"), py::arg("Q"));
}

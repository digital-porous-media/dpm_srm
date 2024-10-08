#ifndef SRM2D_HPP
#define SRM2D_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "SRM.hpp"

namespace py = pybind11;

template <typename T>
class SRM2D : public SRM<T, 2>
{
public:
    // Constructor
    SRM2D(const py::array_t<T> &img, double Q);
    ~SRM2D() {}

    // Get the segmentation result as a 2D array of region labels
    py::array_t<T> getSegmentation() const override;

private:
    const T *img_ptr;
    const int width, height;

    // Initialize each voxel as its own region
    void initializeRegions() override;
    void initializeNeighbors() override;
    void mergeAllNeighbors() override;
    void updateAverages() override;
};

// SRM3D constructor
template <typename T>
SRM2D<T>::SRM2D(const py::array_t<T> &img, double q)
    : SRM<T, 2>(q), width(img.shape(1)), height(img.shape(0))
{
    // Access pointer to np array
    py::buffer_info buf = img.request();

    if (buf.ndim != 2)
    {
        std::cerr << "Expected 2D array, but got " << buf.ndim << std::endl;
        throw std::runtime_error("Error: Expected 2D array"); // Handle the error accordingly
    }

    // Ensure the data type is correct
    if (buf.itemsize != sizeof(T))
    {
        std::cerr << "Expected int data type, but got item size: " << buf.itemsize << std::endl;
        throw std::runtime_error("Error: Incorrect data type"); // Handle the error accordingly
    }

    img_ptr = static_cast<const T *>(buf.ptr);

    if (!img_ptr)
    {
        std::cerr << "img_ptr is null!" << std::endl;
        throw std::runtime_error("Error: img_ptr is null!"); // or handle the error appropriately
    }

    // Initialize region stats
    this->average.resize(width * height, 0.0);
    this->count.resize(width * height, 0);
    this->regionIndex.resize(width * height, -1);

    // Calculate factor and logDelta based on image dimensions
    this->delta = 1.0f / (6 * width * height);            // delta = 1 / (6 * w * h * d)
    this->logDelta = 2.0f * std::log(6 * width * height); // logDelta = 2 * log(6 * w * h * d)
}

// Initialize each voxel as its own region
template <typename T>
void SRM2D<T>::initializeRegions()
{
    const T *pixel = img_ptr;
    for (int i = 0; i < width * height; ++i)
    {
        this->average[i] = pixel[i]; //& 0xff;
        this->count[i] = 1;
        this->regionIndex[i] = i;
    }
}

// Initialize neighbor pairs and bucket sort
template <typename T>
void SRM2D<T>::initializeNeighbors()
{
    // Create a vector to store the neighbors of each voxel
    this->nextNeighbor.resize(2 * width * height);
    this->neighborBucket.resize(static_cast<uint64_t>(this->g), -1);

    // Bucket sort
    // Allocate memory on the heap for nextPixel
    const T *pixel = img_ptr; // pointer to beginning of slice k
    for (int j = height - 1; j >= 0; j--)
    {
        for (int i = width - 1; i >= 0; i--)
        {
            uint64_t index = i + width * j;
            uint64_t neighborIndex = 2 * index;
            // vertical
            if (j < height - 1)
            {
                SRM<T, 2>::addNeighborPair(neighborIndex + 1, pixel, index, index + width);
            }

            // horizontal
            if (i < width - 1)
            {
                SRM<T, 2>::addNeighborPair(neighborIndex, pixel, index, index + 1);
            }
        }
    }
}

// Merge regions based on the predicate criterion
template <typename T>
void SRM2D<T>::mergeAllNeighbors()
{
    uint64_t len = static_cast<uint64_t>(this->g);

    for (uint64_t i = 0; i < len; ++i)
    {
        int64_t neighborIndex = this->neighborBucket[i];

        while (neighborIndex >= 0)
        {
            uint64_t i1 = neighborIndex / 2;
            uint64_t i2 = i1 + (0 == (neighborIndex & 1) ? 1 : width);
            i1 = SRM<T, 2>::getRegionIndex(i1);
            i2 = SRM<T, 2>::getRegionIndex(i2);

            if (i1 != i2 && SRM<T, 2>::predicate(i1, i2))
                SRM<T, 2>::mergeRegions(i1, i2);

            neighborIndex = this->nextNeighbor[neighborIndex];
        }
    }
}

// TODO: Check original code for what this is doing
// template <typename T>
// int SRM3D<T>::consolidateRegions()
// {
//     int len = width * height * depth;
//     int counter = 0;
//     for (int i = 0; i < len; i++)
//     {
//         if (regionIndex[i] < 0)
//         {
//             regionIndex[i] = regionIndex[-1 - regionIndex[i]];
//         }
//         else
//             regionIndex[i] = counter++;
//     }
//     return counter;
// }

// Get the segmentation result as a 3D array of region labels
template <typename T>
void SRM2D<T>::updateAverages()
{
    for (uint64_t i = 0; i < width * height; i++)
    {
        this->average[i] = this->average[SRM<T, 2>::getRegionIndex(i)];
    }
}

template <typename T>
py::array_t<T> SRM2D<T>::getSegmentation() const
{
    // Create an np array for the output
    auto result_array = py::array_t<T>({height, width});
    auto result_buf_info = result_array.request();
    T *result_ptr = static_cast<T *>(result_buf_info.ptr);

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            uint64_t index = i * height + j;
            result_ptr[i * height + j] = static_cast<T>(this->average[index]);
        }
    }
    return result_array;
}

#endif // SRM2D_HPP

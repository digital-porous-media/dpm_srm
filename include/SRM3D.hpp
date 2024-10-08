/*
 * This file is adapted from Statistical Region Merging by Johannes Schindelin.
 * Original code licensed under the BSD 2-Clause License.
 *
 * Copyright (C) 2009 - 2013 Johannes Schindelin.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of any organization.
 */

#ifndef SRM3D_HPP
#define SRM3D_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "SRM.hpp"

namespace py = pybind11;

template <typename T>
class SRM3D : public SRM<T, 3>
{
public:
    // Constructor
    SRM3D(const py::array_t<T> &img, double Q);
    ~SRM3D() {}

    // Get the segmentation result as a 3D array of region labels
    py::array_t<T> getSegmentation() const override;

private:
    const T *img_ptr;
    const int width, height, depth;

    // Initialize each voxel as its own region
    void initializeRegions() override;
    void initializeNeighbors() override;
    void mergeAllNeighbors() override;
    void updateAverages() override;
};

// SRM3D constructor
template <typename T>
SRM3D<T>::SRM3D(const py::array_t<T> &img, double q)
    : SRM<T, 3>(q), width(img.shape(2)), height(img.shape(1)), depth(img.shape(0))
{
    // Access pointer to np array
    py::buffer_info buf = img.request();

    if (buf.ndim != 3)
    {
        std::cerr << "Expected 3D array, but got " << buf.ndim << std::endl;
        throw std::runtime_error("Error: Expected 3D array"); // Handle the error accordingly
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
    this->average.resize(width * height * depth, 0.0);
    this->count.resize(width * height * depth, 0);
    this->regionIndex.resize(width * height * depth, -1);

    // Calculate factor and logDelta based on image dimensions
    this->delta = 1.0f / (6 * width * height * depth);            // delta = 1 / (6 * w * h * d)
    this->logDelta = 2.0f * std::log(6 * width * height * depth); // logDelta = 2 * log(6 * w * h * d)
}

// Initialize each voxel as its own region
template <typename T>
void SRM3D<T>::initializeRegions()
{
    for (int j = 0; j < depth; j++)
    {
        const T *pixel = img_ptr + (j * width * height);
        uint64_t offset = j * width * height;
        for (int i = 0; i < width * height; ++i)
        {
            this->average[offset + i] = pixel[i]; //& 0xff;
            this->count[offset + i] = 1;
            this->regionIndex[offset + i] = offset + i;
        }
    }
}

// Initialize neighbor pairs and bucket sort
template <typename T>
void SRM3D<T>::initializeNeighbors()
{
    // Create a vector to store the neighbors of each voxel
    this->nextNeighbor.resize(3 * width * height * depth);
    this->neighborBucket.resize(static_cast<uint64_t>(this->g), -1);

    // Bucket sort
    // Allocate memory on the heap for nextPixel
    T *nextPixel = new T[width * height]();
    for (int k = depth - 1; k >= 0; k--)
    {
        const T *pixel = img_ptr + (k * width * height); // pointer to beginning of slice k
        for (int j = height - 1; j >= 0; j--)
        {
            for (int i = width - 1; i >= 0; i--)
            {
                uint64_t index = i + width * j;
                uint64_t neighborIndex = 3 * (index + k * width * height);

                // depth
                if (k < depth - 1)
                {
                    SRM<T, 3>::addNeighborPair(neighborIndex + 2, pixel, nextPixel, index);
                }

                // vertical
                if (j < height - 1)
                {
                    SRM<T, 3>::addNeighborPair(neighborIndex + 1, pixel, index, index + width);
                }

                // horizontal
                if (i < width - 1)
                {
                    SRM<T, 3>::addNeighborPair(neighborIndex, pixel, index, index + 1);
                }
            }
        }
        std::copy(pixel, pixel + (width * height), nextPixel);
    }
    delete[] nextPixel; // Free allocated memory
}

// Merge regions based on the predicate criterion
template <typename T>
void SRM3D<T>::mergeAllNeighbors()
{
    uint64_t len = static_cast<uint64_t>(this->g);

    for (uint64_t i = 0; i < len; ++i)
    {
        int64_t neighborIndex = this->neighborBucket[i];

        while (neighborIndex >= 0)
        {
            uint64_t i1 = neighborIndex / 3;
            uint64_t value;
            switch (neighborIndex % 3)
            {
            case 0:
                value = 1;
                break;
            case 1:
                value = width;
                break;
            case 2:
                value = width * height;
                break;
            }
            uint64_t i2 = i1 + value;
            i1 = SRM<T, 3>::getRegionIndex(i1);
            i2 = SRM<T, 3>::getRegionIndex(i2);
            if (i1 != i2 && SRM<T, 3>::predicate(i1, i2))
                SRM<T, 3>::mergeRegions(i1, i2);

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
void SRM3D<T>::updateAverages()
{
    for (uint64_t i = 0; i < width * height * depth; i++)
    {
        this->average[i] = this->average[SRM<T, 3>::getRegionIndex(i)];
    }
}

template <typename T>
py::array_t<T> SRM3D<T>::getSegmentation() const
{
    // Create an np array for the output
    auto result_array = py::array_t<T>({depth, height, width});
    auto result_buf_info = result_array.request();
    T *result_ptr = static_cast<T *>(result_buf_info.ptr);

    // for (int i = 0; i < width; ++i)
    // {
    //     for (int j = 0; j < height; ++j)
    //     {
    //         for (int k = 0; k < depth; k++)
    //         {
    //             uint64_t index = i * height * depth + j * depth + k;                                   // Calculate 1D index
    //             result_ptr[k * height * width + j * width + i] = static_cast<T>(this->average[index]); // Assign the label from the 1D regionIndex
    //         }
    //     }
    // }

    // Iterate through the 3D space and fill the result array
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                uint64_t index = z * height * width + y * width + x; // Adjusted index for 3D-to-1D
                // uint64_t index = x * height * depth + y * depth + z; // 1D index for average
                result_ptr[z * height * width + y * width + x] = static_cast<T>(this->average[index]);
            }
        }
    }
    return result_array;
}

#endif // SRM3D_HPP

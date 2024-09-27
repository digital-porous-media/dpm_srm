#ifndef SRM3D_HPP
#define SRM3D_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

namespace py = pybind11;

template <typename T>
class SRM3D
{
public:
    // Constructor
    SRM3D(const py::array_t<T> &img, double Q);
    // Perform the segmentation
    void segment();

    // Get the segmentation result as a 3D array of region labels
    py::array_t<T> getSegmentation() const;

private:
    const T *img_ptr;

    // py::array_t<int> image;
    // std::vector<std::vector<std::vector<int>>> region_labels;
    // std::vector<Region> regions;
    std::vector<uint64_t> nextNeighbor;
    std::vector<int64_t> neighborBucket;
    std::vector<double> average;
    std::vector<uint64_t> count;
    std::vector<int64_t> regionIndex;

    const int width, height, depth;

    const double g = static_cast<double>(std::numeric_limits<T>::max() + 1); // Some constant
    float logDelta;                                                          // Log delta for merging criterion
    float Q;                                                                 // Parameter Q
    float delta;                                                             // Parameter delta
    float factor;                                                            // Merging factor

    // Initialize each voxel as its own region
    void initializeRegions();

    void addNeighborPair(int neighborID, const T *pixel, T *nextPixel, int i);
    void addNeighborPair(int neighborID, const T *pixel, int i, int j);

    void initializeNeighbors();

    int64_t getRegionIndex(int64_t i);

    // Check if two regions should be merged based on the new criteria
    bool predicate(int64_t i1, int64_t i2) const;

    // Merge two regions
    void mergeRegions(int64_t label1, int64_t label2);

    // Core SRM logic: Merge regions based on the new criterion
    void mergeAllNeighbors3D();

    // int consolidateRegions();
};

template <typename T>
// SRM3D constructor
SRM3D<T>::SRM3D(const py::array_t<T> &img, double q)
    : Q(q), width(img.shape(2)), height(img.shape(1)), depth(img.shape(0))
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
    average.resize(width * height * depth, 0.0);
    count.resize(width * height * depth, 0);
    regionIndex.resize(width * height * depth, -1);

    // Calculate factor and logDelta based on image dimensions
    delta = 1.0f / (6 * width * height * depth);            // delta = 1 / (6 * w * h * d)
    factor = (g * g) / (2 * Q);                             // factor = g^2 / 2Q
    logDelta = 2.0f * std::log(6 * width * height * depth); // logDelta = 2 * log(6 * w * h * d)
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
            average[offset + i] = pixel[i]; //& 0xff;
            count[offset + i] = 1;
            regionIndex[offset + i] = offset + i;
        }
    }
}

// Initialize neighbor pairs and bucket sort
template <typename T>
void SRM3D<T>::initializeNeighbors()
{
    // Create a vector to store the neighbors of each voxel
    nextNeighbor.resize(3 * width * height * depth);
    neighborBucket.resize(static_cast<uint64_t>(g), -1);

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
                int index = i + width * j;
                int neighborIndex = 3 * (index + k * width * height);

                // depth
                if (k < depth - 1)
                {
                    addNeighborPair(neighborIndex + 2, pixel, nextPixel, index);
                }

                // vertical
                if (j < height - 1)
                {
                    addNeighborPair(neighborIndex + 1, pixel, index, index + width);
                }

                // horizontal
                if (i < width - 1)
                {
                    addNeighborPair(neighborIndex, pixel, index, index + 1);
                }
            }
        }
        std::copy(pixel, pixel + (width * height), nextPixel);
    }
    delete[] nextPixel; // Free allocated memory
}

// Function to add neighbor pair to bucket
template <typename T>
void SRM3D<T>::addNeighborPair(int neighborID, const T *pixel, T *nextPixel, int i)
{
    auto difference = std::abs(pixel[i] - nextPixel[i]);
    nextNeighbor[neighborID] = neighborBucket[difference];
    neighborBucket[difference] = neighborID;
}

// Overloaded function to add neighbor pair to bucket
template <typename T>
void SRM3D<T>::addNeighborPair(int neighborID, const T *pixel, int i, int j)
{
    int difference = abs(pixel[i] - pixel[j]);
    nextNeighbor[neighborID] = neighborBucket[difference];
    neighborBucket[difference] = neighborID;
}

// Get the region label index recursively
template <typename T>
int64_t SRM3D<T>::getRegionIndex(int64_t i)
{
    i = regionIndex[i];
    while (i < 0)
        i = regionIndex[-1 - i];
    return i;
}

// Check if two regions should be merged based on the new criteria
template <typename T>
bool SRM3D<T>::predicate(int64_t i1, int64_t i2) const
{
    double difference = average[i1] - average[i2];
    double log1 = log(1.0f + count[i1]) * (g < count[i1] ? g : count[i1]);
    double log2 = log(1.0f + count[i2]) * (g < count[i2] ? g : count[i2]);

    return difference * difference <
           .1f * factor * ((log1 + logDelta) / count[i1] + ((log2 + logDelta) / count[i2]));
}

// Merge two regions
template <typename T>
void SRM3D<T>::mergeRegions(int64_t i1, int64_t i2)
{
    if (i1 == i2)
        return;
    int64_t mergedCount = count[i1] + count[i2];
    double mergedAverage = (average[i1] * count[i1] + average[i2] * count[i2]) / mergedCount;

    // merge larger index into smaller index
    if (i1 > i2)
    {
        average[i2] = mergedAverage;
        count[i2] = mergedCount;
        regionIndex[i1] = -1 - i2;
    }
    else
    {
        average[i1] = mergedAverage;
        count[i1] = mergedCount;
        regionIndex[i2] = -1 - i1;
    }
}

// Merge regions based on the predicate criterion
template <typename T>
void SRM3D<T>::mergeAllNeighbors3D()
{
    uint64_t len = static_cast<uint64_t>(g);

    for (uint64_t i = 0; i < len; ++i)
    {
        int64_t neighborIndex = neighborBucket[i];

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
            i1 = getRegionIndex(i1);
            i2 = getRegionIndex(i2);
            if (i1 != i2 && predicate(i1, i2))
                mergeRegions(i1, i2);

            neighborIndex = nextNeighbor[neighborIndex];
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

// Perform the segmentation
template <typename T>
void SRM3D<T>::segment()
{
    std::cout << "Initializing regions... " << std::endl;
    initializeRegions();
    std::cout << "Initializing neighbors... " << std::endl;
    initializeNeighbors();
    std::cout << "Merging... " << std::endl;
    mergeAllNeighbors3D();
    for (uint64_t i = 0; i < width * height * depth; i++)
    {
        average[i] = average[getRegionIndex(i)];
    }
}

// Get the segmentation result as a 3D array of region labels
template <typename T>
py::array_t<T> SRM3D<T>::getSegmentation() const
{
    // Create an np array for the output
    auto result_array = py::array_t<T>({depth, height, width});
    auto result_buf_info = result_array.request();
    T *result_ptr = static_cast<T *>(result_buf_info.ptr);

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            for (int k = 0; k < depth; ++k)
            {
                uint64_t index = i * height * depth + j * depth + k;             // Calculate 1D index
                result_ptr[i * height * width + j * width + k] = average[index]; // Assign the label from the 1D regionIndex
            }
        }
    }
    return result_array;
}

#endif // SRM3D_HPP

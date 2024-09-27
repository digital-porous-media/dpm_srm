#ifndef SRM_HPP
#define SRM_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

namespace py = pybind11;

template <typename T, int Dimensions>
class SRM
{
public:
    // Constructor
    SRM(const double Q);

    // Destructor
    virtual ~SRM() {}

    // Perform the segmentation
    virtual void segment();

    // Get the segmentation result as an array of region labels
    virtual py::array_t<T> getSegmentation() const = 0;

protected:
    double Q; // Parameter Q
    double g; // Some constant
    double factor;
    float delta, logDelta;

    std::vector<uint64_t> nextNeighbor;
    std::vector<int64_t> neighborBucket;
    std::vector<double> average;
    std::vector<uint64_t> count;
    std::vector<int64_t> regionIndex;

    // Initialize each voxel as its own region
    virtual void initializeRegions() = 0;

    virtual void initializeNeighbors() = 0;
    void addNeighborPair(int neighborID, const T *pixel, T *nextPixel, int i);
    void addNeighborPair(int neighborID, const T *pixel, int i, int j);

    int64_t getRegionIndex(int64_t i);

    // Check if two regions should be merged based on the new criteria
    virtual bool predicate(int64_t i1, int64_t i2) const;

    // Merge two regions
    void mergeRegions(int64_t label1, int64_t label2);

    // Merge regions based on the new criterion. Force derived classes to implement this.
    virtual void mergeAllNeighbors() = 0;

    virtual void updateAverages() = 0;

    // int consolidateRegions();
};

// Constructor
template <typename T, int Dimensions>
SRM<T, Dimensions>::SRM(double Q)
    : Q(Q), g(static_cast<double>(std::numeric_limits<T>::max() + 1)), factor((g * g) / (2 * Q)) {}

// Function to add neighbor pair to bucket
template <typename T, int Dimensions>
void SRM<T, Dimensions>::addNeighborPair(int neighborID, const T *pixel, T *nextPixel, int i)
{
    auto difference = std::abs(pixel[i] - nextPixel[i]);
    nextNeighbor[neighborID] = neighborBucket[difference];
    neighborBucket[difference] = neighborID;
}

// Overloaded function to add neighbor pair to bucket
template <typename T, int Dimensions>
void SRM<T, Dimensions>::addNeighborPair(int neighborID, const T *pixel, int i, int j)
{
    int difference = std::abs(pixel[i] - pixel[j]);
    nextNeighbor[neighborID] = neighborBucket[difference];
    neighborBucket[difference] = neighborID;
}

// Get the region label index recursively
template <typename T, int Dimensions>
int64_t SRM<T, Dimensions>::getRegionIndex(int64_t i)
{
    i = regionIndex[i];
    while (i < 0)
        i = regionIndex[-1 - i];
    return i;
}

// Check if two regions should be merged based on the new criteria
template <typename T, int Dimensions>
bool SRM<T, Dimensions>::predicate(int64_t i1, int64_t i2) const
{
    double difference = average[i1] - average[i2];
    double log1 = log(1.0f + count[i1]) * (g < count[i1] ? g : count[i1]);
    double log2 = log(1.0f + count[i2]) * (g < count[i2] ? g : count[i2]);

    return difference * difference <
           .1f * factor * ((log1 + logDelta) / count[i1] + ((log2 + logDelta) / count[i2]));
}

// Merge two regions
template <typename T, int Dimensions>
void SRM<T, Dimensions>::mergeRegions(int64_t i1, int64_t i2)
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

// Perform the segmentation
template <typename T, int Dimensions>
void SRM<T, Dimensions>::segment()
{
    initializeRegions();
    initializeNeighbors();
    mergeAllNeighbors();
    updateAverages();
}

#endif // SRM_HPP

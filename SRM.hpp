#ifndef SRM3D_HPP
#define SRM3D_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

class SRM3D
{
public:
    // Constructor
    SRM3D(const py::array_t<uint16_t> &img, float Q);
    // Perform the segmentation
    void segment();

    // Get the segmentation result as a 3D array of region labels
    py::array_t<uint16_t> getSegmentation() const;

private:
    const uint16_t *img_ptr;

    // py::array_t<int> image;
    // std::vector<std::vector<std::vector<int>>> region_labels;
    // std::vector<Region> regions;
    std::vector<int> nextNeighbor;
    std::vector<int> neighborBucket;
    std::vector<double> average;
    std::vector<int> count;
    std::vector<int> regionIndex;

    const int width, height, depth;

    const float g = 65535.0; // Some constant
    float logDelta;          // Log delta for merging criterion
    float Q;                 // Parameter Q
    float delta;             // Parameter delta
    float factor;            // Merging factor

    // Initialize each voxel as its own region
    void initializeRegions();

    void addNeighborPair(int neighborID, const uint16_t *pixel, uint16_t *nextPixel, int i);
    void addNeighborPair(int neighborID, const uint16_t *pixel, int i, int j);

    void initializeNeighbors();

    int getRegionIndex(int i);

    // Check if two regions should be merged based on the new criteria
    bool predicate(int i1, int i2) const;

    // Merge two regions
    void mergeRegions(int label1, int label2);

    // Core SRM logic: Merge regions based on the new criterion
    void mergeAllNeighbors3D();

    int consolidateRegions();
};

#endif // SRM3D_HPP

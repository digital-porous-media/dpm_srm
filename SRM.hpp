#ifndef SRM3D_HPP
#define SRM3D_HPP

#include <windows.h>
#include <vector>
#include <cmath>

class SRM3D {
public:
    // Constructor
    SRM3D(const std::vector<std::vector<std::vector<int>>>& img, double Q);
    // Perform the segmentation
    void segment();

    // Get the segmentation result as a 3D array of region labels
    std::vector<std::vector<std::vector<int>>> getSegmentation() const;


private:
    // struct Region {
    //     int label;
    //     double intensity_sum;
    //     int voxel_count;

    //     // Constructor
    //     Region(int l, double intensity, int count);
        
    //     // Compute the average intensity of the region
    //     double average_intensity() const;

    //     // Merge another region into this region
    //     void merge(const Region& other);
    // };

    std::vector<std::vector<std::vector<int>>> image;
    std::vector<std::vector<std::vector<int>>> region_labels;
    // std::vector<Region> regions;
    std::vector<int> nextNeighbor;
    std::vector<int> neighborBucket;
    std::vector<float> average;
    std::vector<int> count;
    std::vector<int> regionIndex;

    const int width, height, depth;

    const double g = 256;  // Some constant
    float logDelta;   // Log delta for merging criterion
    float Q;          // Parameter Q
    float delta;      // Parameter delta
    float factor;      // Merging factor


    // Initialize each voxel as its own region
    void initializeRegions();

    void addNeighborPair(int neighborID, const std::vector<std::vector<int>>& pixel, const std::vector<std::vector<int>>& nextPixel, int i);
    void addNeighborPair(int neighborID, const std::vector<std::vector<int>>& pixel, int i, int j);

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

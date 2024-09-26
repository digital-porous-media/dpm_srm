#ifndef SRM3D_HPP
#define SRM3D_HPP

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
    struct Region {
        int label;
        double intensity_sum;
        int voxel_count;

        // Constructor
        Region(int l, double intensity, int count);
        
        // Compute the average intensity of the region
        double average_intensity() const;

        // Merge another region into this region
        void merge(const Region& other);
    };

    std::vector<std::vector<std::vector<int>>> image;
    std::vector<std::vector<std::vector<int>>> region_labels;
    std::vector<Region> regions;
    int width, height, depth;

    double factor;      // Merging factor
    const double g = 256;  // Some constant
    double logDelta;   // Log delta for merging criterion
    double Q;          // Parameter Q
    double delta;      // Parameter delta


    // Initialize each voxel as its own region
    void initializeRegions();

    // Check if two regions should be merged based on the new criteria
    bool predicate(int i1, int i2) const;

    // Merge two regions
    void mergeRegions(int label1, int label2);

    // Core SRM logic: Merge regions based on the new criterion
    void mergeProcess();
};

#endif // SRM3D_HPP

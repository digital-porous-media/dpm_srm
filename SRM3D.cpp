#include "SRM3D.hpp"
#include <iostream>
#include <cmath>

// Region constructor
SRM3D::Region::Region(int l, double intensity, int count)
    : label(l), intensity_sum(intensity), voxel_count(count) {}

// Compute the average intensity of the region
double SRM3D::Region::average_intensity() const {
    return intensity_sum / voxel_count;
}

// Merge another region into this region
void SRM3D::Region::merge(const Region& other) {
    intensity_sum += other.intensity_sum;
    voxel_count += other.voxel_count;
}

// SRM3D constructor
SRM3D::SRM3D(const std::vector<std::vector<std::vector<int>>>& img, double q)
    : image(img), Q(q), width(img.size()), height(img[0].size()), depth(img[0][0].size()) 
      {
        // Calculate factor and logDelta based on image dimensions
        factor = (g * g) / (2 * Q); // factor = g^2 / 2Q
        logDelta = 2.0 * std::log(6 * width * height * depth); // logDelta = 2 * log(6 * w * h * d)
      }


// Initialize each voxel as its own region
void SRM3D::initializeRegions() {
    int label_counter = 0;
    region_labels.resize(width, std::vector<std::vector<int>>(height, std::vector<int>(depth)));
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < depth; ++k) {
                region_labels[i][j][k] = label_counter;
                regions.push_back(Region(label_counter, image[i][j][k], 1));
                label_counter++;
            }
        }
    }
}

// Check if two regions should be merged based on the new criteria
bool SRM3D::predicate(int i1, int i2) const {
    double difference = regions[i1].average_intensity() - regions[i2].average_intensity();

    double log1 = std::log(1 + regions[i1].voxel_count) * std::min(g, static_cast<double>(regions[i1].voxel_count));
    double log2 = std::log(1 + regions[i2].voxel_count) * std::min(g, static_cast<double>(regions[i2].voxel_count));
    double mergeCriterion = 0.1 * factor * ((log1 + logDelta) / regions[i1].voxel_count + ((log2 + logDelta) / regions[i2].voxel_count));
    return (difference * difference < mergeCriterion);
}

// Merge two regions
void SRM3D::mergeRegions(int label1, int label2) {
    Region& r1 = regions[label1];
    Region& r2 = regions[label2];
    if (label1 != label2 && predicate(label1, label2)) {
        r1.merge(r2);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                for (int k = 0; k < depth; ++k) {
                    if (region_labels[i][j][k] == label2) {
                        region_labels[i][j][k] = label1;
                    }
                }
            }
        }
    }
}

// Merge regions based on the predicate criterion
void SRM3D::mergeProcess() {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < depth; ++k) {
                int current_label = region_labels[i][j][k];
                if (i + 1 < width) mergeRegions(current_label, region_labels[i + 1][j][k]);
                if (i - 1 >= 0) mergeRegions(current_label, region_labels[i - 1][j][k]);
                if (j + 1 < height) mergeRegions(current_label, region_labels[i][j + 1][k]);
                if (j - 1 >= 0) mergeRegions(current_label, region_labels[i][j - 1][k]);
                if (k + 1 < depth) mergeRegions(current_label, region_labels[i][j][k + 1]);
                if (k - 1 >= 0) mergeRegions(current_label, region_labels[i][j][k - 1]);
            }
        }
    }
}

// Perform the segmentation
void SRM3D::segment() {
    initializeRegions();
    // mergeProcess();
}


// Get the segmentation result as a 3D array of region labels
std::vector<std::vector<std::vector<int>>> SRM3D::getSegmentation() const {
    return region_labels;
}

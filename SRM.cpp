#include "SRM.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// // Region constructor
// SRM3D::Region::Region(int l, double intensity, int count)
//     : label(l), intensity_sum(intensity), voxel_count(count) {}

// // Compute the average intensity of the region
// double SRM3D::Region::average_intensity() const {
//     return intensity_sum / voxel_count;
// }

// // Merge another region into this region
// void SRM3D::Region::merge(const Region& other) {
//     intensity_sum += other.intensity_sum;
//     voxel_count += other.voxel_count;
// }

// SRM3D constructor
SRM3D::SRM3D(const std::vector<std::vector<std::vector<int>>> &img, double q)
    : image(img), Q(q), width(img.size()), height(img[0].size()), depth(img[0][0].size())
{
    // Calculate factor and logDelta based on image dimensions
    delta = 1.0f / (6 * width * height * depth);            // delta = 1 / (6 * w * h * d)
    factor = (g * g) / (2 * Q);                             // factor = g^2 / 2Q
    logDelta = 2.0f * std::log(6 * width * height * depth); // logDelta = 2 * log(6 * w * h * d)
}

// Initialize each voxel as its own region
void SRM3D::initializeRegions()
{
    int label_counter = 0;
    region_labels.resize(width, std::vector<std::vector<int>>(height, std::vector<int>(depth)));
    average.resize(width * height * depth, 0.0);
    count.resize(width * height * depth, 0);
    regionIndex.resize(width * height * depth, -1);

    for (int j = 0; j < depth; ++j)
    {
        const auto &pixel = image[j];
        int offset = j * width * height;
        for (int i = 0; i < width * height; ++i)
        {
            int row = i / width;
            int col = i % width;
            average[offset + i] = pixel[row][col] & 0xff;
            count[offset + i] = 1;
            regionIndex[offset + i] = offset + i;
        }
    }
}

void SRM3D::initializeNeighbors()
{

    // Create a vector to store the neighbors of each voxel
    nextNeighbor.resize(3 * width * height * depth);
    neighborBucket.resize(static_cast<int>(g), -1);

    // Bucket sort
    std::vector<std::vector<int>> nextPixel(height, std::vector<int>(width, 0));

    for (int k = depth - 1; k >= 0; k--)
    {
        const auto &pixel = image[k];
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
        nextPixel = pixel;
    }
}

void SRM3D::addNeighborPair(int neighborID, const std::vector<std::vector<int>> &pixel, const std::vector<std::vector<int>> &nextPixel, int i)
{
    int row = i / width;
    int col = i % width;

    int difference = std::abs(static_cast<int>(pixel[row][col]) - static_cast<int>(nextPixel[row][col]));
    nextNeighbor[neighborID] = neighborBucket[difference];
    neighborBucket[difference] = neighborID;
}

void SRM3D::addNeighborPair(int neighborID, const std::vector<std::vector<int>> &pixel, int i, int j)
{
    int row_i = i / width;
    int col_i = i % width;

    int row_j = j / width;
    int col_j = j % width;
    int difference = abs(static_cast<int>(pixel[row_i][col_i]) - static_cast<int>(pixel[row_j][col_j]));
    nextNeighbor[neighborID] = neighborBucket[difference];
    neighborBucket[difference] = neighborID;
}

int SRM3D::getRegionIndex(int i)
{
    i = regionIndex[i];
    while (i < 0)
        i = regionIndex[-1 - i];
    return i;
}

// Check if two regions should be merged based on the new criteria
bool SRM3D::predicate(int i1, int i2) const
{
    double difference = average[i1] - average[i2];
    double log1 = log(1 + count[i1]) * (g < count[i1] ? g : count[i1]);
    double log2 = log(1 + count[i2]) * (g < count[i2] ? g : count[i2]);

    return difference * difference <
           .1f * factor * ((log1 + logDelta) / count[i1] + ((log2 + logDelta) / count[i2]));
    // double log1 = std::log(1 + static_cast<double>(count[i1])) * std::min(g, static_cast<double>(count[i1]));
    // double log2 = std::log(1 + static_cast<double>(count[i2])) * std::min(g, static_cast<double>(count[i2]));
    // double mergeCriterion = 0.1 * factor * ((log1 + logDelta) / regions[i1].voxel_count + ((log2 + logDelta) / regions[i2].voxel_count));
    // return (difference * difference < mergeCriterion);
}

// Merge two regions
void SRM3D::mergeRegions(int i1, int i2)
{
    if (i1 == i2)
        return;
    int mergedCount = count[i1] + count[i2];
    float mergedAverage = (average[i1] * count[i1] + average[i2] * count[i2]) / mergedCount;

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
void SRM3D::mergeAllNeighbors3D()
{
    int len = (int)g;

    for (int i = 0; i < len; i++)
    {
        int neighborIndex = neighborBucket[i];
        while (neighborIndex >= 0)
        {
            int i1 = neighborIndex / 3;
            int i2 = i1 + (0 == (neighborIndex % 3) ? 1 : (1 == (neighborIndex % 3) ? width : width * height));

            i1 = getRegionIndex(i1);
            i2 = getRegionIndex(i2);

            if (i1 != i2 && predicate(i1, i2))
                mergeRegions(i1, i2);

            neighborIndex = nextNeighbor[neighborIndex];
        }
    }
}

int SRM3D::consolidateRegions()
{
    int len = width * height * depth;
    int counter = 0;
    for (int i = 0; i < len; i++)
    {
        if (regionIndex[i] < 0)
        {
            regionIndex[i] = regionIndex[-1 - regionIndex[i]];
        }
        else
            regionIndex[i] = counter++;
    }
    return counter;
}

// Perform the segmentation
void SRM3D::segment()
{
    initializeRegions();
    // mergeProcess();
    initializeNeighbors();
    mergeAllNeighbors3D();
    for (int i = 0; i < width * height * depth; i++)
    {
        average[i] = average[getRegionIndex(i)];
    }
    region_labels.resize(width, std::vector<std::vector<int>>(height, std::vector<int>(depth)));

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            for (int k = 0; k < depth; ++k)
            {
                int index = i * height * depth + j * depth + k; // Calculate 1D index
                region_labels[i][j][k] = average[index];        // Assign the label from the 1D regionIndex
            }
        }
    }
}

// Get the segmentation result as a 3D array of region labels
std::vector<std::vector<std::vector<int>>> SRM3D::getSegmentation() const
{
    return region_labels;
}
#include <iostream>
#include <vector>
#include "SRM3D.hpp" // Make sure to include the SRM3D header

void displaySegmentation(const std::vector<std::vector<std::vector<int>>>& segmentation) {
    for (size_t i = 0; i < segmentation.size(); ++i) {
        for (size_t j = 0; j < segmentation[i].size(); ++j) {
            for (size_t k = 0; k < segmentation[i][j].size(); ++k) {
                std::cout << "Voxel(" << i << "," << j << "," << k << ") -> Region: "
                          << segmentation[i][j][k] << std::endl;
            }
        }
    }
}


// Main function for testing
int main() {
    // Example 3D image (3x3x3) with random values
    std::vector<std::vector<std::vector<int> > > image = {
        {
            {1, 2, 1},
            {2, 3, 2},
            {1, 2, 1}
        },
        {
            {1, 2, 1},
            {2, 3, 2},
            {1, 2, 1}
        },
        {
            {1, 2, 1},
            {2, 3, 2},
            {1, 2, 1}
        }
    };
    std::vector<std::vector<std::vector<int> > > segmentation;

    // Parameters for the SRM algorithm
    double g = 3.0; // Example value for g
    double Q = 3.0; // Example value for Q

    SRM3D srm(image, g, Q);
    srm.segment();
    segmentation = srm.getSegmentation();

    // Display the segmentation
    displaySegmentation(segmentation);

    return 0;
}

import numpy as np
from build import srm3d

# Create a sample 3D image (for example, a 2x2x2 array)
image = np.random.randint(0, 256, size=(100, 100, 100), dtype=np.uint8)
image.tofile("test_image.raw")

# # Set a threshold for merging
# threshold = 1.0

# # Create an instance of the SRM3D class
# srm = srm3d.SRM3D(image=image, Q=5)

# # Perform segmentation
# srm.segment()

# # Get the result
# segmentation_result = srm.get_result()


# print(segmentation_result)

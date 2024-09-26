import numpy as np
from build import srm3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import perf_counter_ns

# Create a sample 3D image (for example, a 2x2x2 array)
image = np.random.randint(0, 256, size=(200, 200, 200), dtype=np.uint8)
plt.imshow(image[50], cmap="Greys_r")
plt.colorbar()
image.tofile("test_image.raw")

# # Set a threshold for merging
# threshold = 1.0

tick = perf_counter_ns()
# Create an instance of the SRM3D class
srm = srm3d.SRM3D(image=image, Q=5)

# Perform segmentation
srm.segment()
print(f"{(perf_counter_ns() - tick) * 1e-9 : .4f}s")
# Get the result
segmentation_result = np.asarray(srm.get_result()).reshape(image.shape)


plt.figure()
plt.imshow(segmentation_result[50], cmap="Greys_r")
plt.colorbar()
plt.show()


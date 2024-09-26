import numpy as np
import ctypes
from build import srm3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import perf_counter_ns
np.random.seed(130621)

image = np.random.randint(0, 65535, size=(100, 100, 100), dtype=np.uint16)

plt.imshow(image[0], cmap="Greys_r")
plt.colorbar()
# image.tofile("test_image.raw")

# Start a timer
tick = perf_counter_ns()

# Create an instance of the SRM3D class
srm = srm3d.SRM3D(image, Q=5)

# Perform segmentation
srm.segment()
print(f"{(perf_counter_ns() - tick) * 1e-9 : .4f}s")

# Get the result
segmentation_result = srm.get_result()


plt.figure()
plt.imshow(segmentation_result[0], cmap="Greys_r")
plt.colorbar()
plt.show()


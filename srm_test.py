import numpy as np
import ctypes
from build.srm3d import SRM3D_u1, SRM3D_u2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import perf_counter_ns
np.random.seed(130621)

image_8 = np.random.randint(0, 256, size=(100, 100, 100), dtype=np.uint8)
image_16 = np.random.randint(0, 65536, size=(100, 100, 100), dtype=np.uint16)

plt.imshow(image_8[50], cmap="Greys_r")
plt.colorbar()

plt.figure()
plt.imshow(image_16[50], cmap="Greys_r")
plt.colorbar()
# image.tofile("test_image.raw")

# Start a timer
tick = perf_counter_ns()

# Create an instance of the SRM3D class
srm8 = SRM3D_u1(image_8, Q=5)

# Perform segmentation
srm8.segment()
print(f"{(perf_counter_ns() - tick) * 1e-9 : .4f}s")


# Start a timer
tick = perf_counter_ns()

# Create an instance of the SRM3D class
srm16 = SRM3D_u2(image_16, Q=15)

# Perform segmentation
srm16.segment()
print(f"{(perf_counter_ns() - tick) * 1e-9 : .4f}s")

# Get the results
segmentation_result_8 = srm8.get_result()
segmentation_result_16 = srm16.get_result()

plt.figure()
plt.imshow(segmentation_result_8[50], cmap="Greys_r")
plt.colorbar()


plt.figure()
plt.imshow(segmentation_result_16[50], cmap="Greys_r")
plt.colorbar()
plt.show()


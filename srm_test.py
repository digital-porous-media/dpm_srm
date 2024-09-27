import numpy as np
import ctypes
from build.dpm_srm import SRM2D_u8, SRM3D_u16, SRM3D_u8
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import perf_counter_ns
np.random.seed(130621)

image_u16 = np.random.randint(0, 256, size=(100, 200, 200), dtype=np.uint16)
# image_8_2d =np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
# image_16 = np.random.randint(0, 65536, size=(100, 100, 100), dtype=np.uint16)

plt.imshow(image_u16[2], cmap="Greys_r")
plt.colorbar()

print("Starting segmentation...")

# Start a timer
tick = perf_counter_ns()

# Create an instance of the SRM3D class
srm16 = SRM3D_u16(image_u16, Q=5.0)
# srm8_2d = SRM2D(image_8_2d, Q=5.0)

# Perform segmentation
srm16.segment()
# srm8_2d.segment()
print(f"{(perf_counter_ns() - tick) * 1e-9 : .4f}s")


# Get the results
segmentation_result_16 = srm16.get_result()
# segmentation_result_8_2d = srm8_2d.get_result()

plt.figure()
plt.imshow(segmentation_result_16[2], cmap="Greys_r")
plt.colorbar()

# plt.figure()
# plt.imshow(segmentation_result_8_2d, cmap="Greys_r")
# plt.colorbar()

plt.show()


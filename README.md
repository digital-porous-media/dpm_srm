# Statistical Region Merging

Statistical Region Merging (SRM) is a method for image segmentation. The labeled image is initialized such that each pixel (or voxel) corresponds to one region. A statistical test on neighboring regions determines whether the mean intensities are similar enough to be merged.

This library is an adaptation of the [SRM plugin to Fiji/ImageJ](https://imagej.net/plugins/statistical-region-merging) and is based on the algorithm proposed in [Nock and Nielsen (2004)](10.1109/tpami.2004.110). Our contribution was to translate the original algorithm to C++ and wrap it in Python. We provide this package under the Digital Porous Media (DPM) organization.


## Installation:
dpm_srm is packaged on [pypi](https://pypi.org/project/dpm-srm/) and can be installed with pip.
```pip install dpm-srm```.

If installing from source, this package requires a C++ compiler.

## Usage Example:
---
This implementation of SRM expects a 2D or 3D grayscale (single color channel) image of type uint8, uint16, or uint32 and a value for *Q*, which is used as a merging criterion. Roughly speaking, *Q* is an estimate of the number of expected regions, though this is not strictly adhered to. The larger the *Q* value, the more regions are produced. The algorithm will return a labeled image of the same shape and datatype as the input image. 

Note that the algorithm performs bucket sorting, where the number of buckets correspond to the maximum allowable value for the particular datatype. Therefore, it's important that intensity values of the input image are scaled over the entire range of the datatype. For example, if the input image is uint8, the image should be scaled such that the minimum intensity value is 0, and the maximum is 255. If the input image is uint16 or uint32, the minimum values should be 0 and the maximum should be 65535 (or 4294967295) respectively.

We wrapped each version (2D vs. 3D, dtype) of the template class into individual class instances. The nomenclature is: SRM[2(or 3)]D_u[number_of_bits]() (e.g. ```SRG2D_u8()```, ```SRG3D_u32()```).

**Python Example:**
```
import dpm_srm
import numpy as np

np.random.seed(130621)
image = np.random.randint(0, 256, size=(100, 200, 200), dtype=np.uint8)

srm_obj = dpm_srm.SRM3D_u8(image, Q=5.0)
srm_obj.segment()
segmentation = srm_obj.get_result()
```


## Acknowledgements
This project includes code adapted from Statistical Region Merging by Johannes Schindelin, which is licensed under the BSD 2-Clause License.

The original code can be found [here](https://github.com/fiji/Statistical_Region_Merging/tree/master)
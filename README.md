# Statistical Region Merging

Installation instructions:
```pip install dpm_srm```

The package has been updated with a Python getter function to simplify usage.

Usage Example:
---
```
import dpm_srm
import numpy as np

np.random.seed(130621)
image = np.random.randint(0, 256, size=(100, 200, 200), dtype=np.uint16)

segmented_image = dpm_srm.segment(image, Q=5.0, rescale=True)
```


## Acknowledgements
This project includes code adapted from Statistical Region Merging by Johannes Schindelin, which is licensed under the BSD 2-Clause License.

The original code can be found [here](https://github.com/fiji/Statistical_Region_Merging/tree/master)

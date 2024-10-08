from ._dpm_srm import SRM2D_u8, SRM2D_u16, SRM2D_u32, SRM3D_u8, SRM3D_u16, SRM3D_u32
import numpy as np


def segment(image: np.ndarray, Q: float = 25., rescale: bool = True) -> np.ndarray:
    """
    Statistical Region Merging Segmentation as implemented in <NAME> et al.

    Parameters:
    ---
    image: A 2D or 3D Numpy array of the image to be segmented
    Q: Used as part of the merging criterion. Larger Q -> more regions. Default: 25.
    rescale: Rescale the image to fit [0-dtype::max()]. This will provide better segmentation results. Default: True

    Returns:
    ---
    segmented_image: A 2D or 3D Numpy array of the image with each region labelled.
    
    """
    srm_dtypes = ["uint8", "uint16", "uint32", "uint64"]
    assert image.ndim == 2 or image.ndim == 3, f"Image must be 2D or 3D, not {image.ndim}D."
    assert str(image.dtype) in srm_dtypes, f"Image must be of type {srm_dtypes}."

    if rescale:
        original_dtype = image.dtype
        original_bytes = image.itemsize
        try:
            dtype_min = np.iinfo(original_dtype).min
            bytes_max = 2**(original_bytes * 8) - 1
        
        except ValueError:
            dtype_min = np.finfo(original_dtype).min
            bytes_max = 2**(original_bytes * 8) - 1
            
        min_val = np.percentile(image, 1)
        max_val = np.percentile(image, 99)
        image[image < min_val] = min_val
        image[image > max_val] = max_val   
        image = (image - min_val) / (max_val - min_val) * bytes_max + dtype_min
        image = image.astype(original_dtype)

    srm_funcs = {"2": {"uint8": SRM2D_u8,
                       "uint16": SRM2D_u16,
                       "uint32": SRM2D_u32},
                "3": {"uint8": SRM3D_u8,
                      "uint16": SRM3D_u16,
                      "uint32": SRM3D_u32}}
    srm_func = srm_funcs[str(image.ndim)][str(image.dtype)]
    srm_obj = srm_func(image, Q)
    srm_obj.segment()
    segmentation = srm_obj.get_result()

    return segmentation

# if __name__ == "__main__":
#     import tifffile
#     import matplotlib.pyplot as plt
#     image = tifffile.imread("/mnt/d/Petrobras_Data/uCT/Depth_1/F4635H_CLEAN_P_13000nm_small.tif")
#     image = image[:50, :50, :50]
#     image = image.astype(np.uint32)

#     # image = image.astype(np.float32)
#     # min_val = np.percentile(image, 1)
#     # max_val = np.percentile(image, 99)
#     # image[image < min_val] = min_val
#     # image[image > max_val] = max_val   
#     # image = (image - min_val) / (max_val - min_val) * 255 - 128
#     # image = image.astype(np.uint8)
#     # rescale = False

#     # if rescale:
#     #     original_dtype = image.dtype
#     #     original_bytes = image.itemsize
#     #     try:
#     #         dtype_min = np.iinfo(original_dtype).min
#     #         bytes_max = 2**(original_bytes * 8) - 1
        
#     #     except ValueError:
#     #         dtype_min = np.finfo(original_dtype).min
#     #         bytes_max = 2**(original_bytes * 8) - 1
        
#     #     print(dtype_min, bytes_max)
#     #     min_val = np.percentile(image, 1)
#     #     max_val = np.percentile(image, 99)
#     #     image[image < min_val] = min_val
#     #     image[image > max_val] = max_val   
#     #     image = (image - min_val) / (max_val - min_val) * bytes_max + dtype_min
#     #     image = image.astype(original_dtype)

    

#     # print(np.amin(image), np.amax(image), image.dtype)

#     segmentation = srm(image, Q=25, rescale=True)

#     plt.figure()
#     plt.imshow(image[20], cmap="Greys_r")
#     plt.colorbar()


#     plt.figure()
#     plt.imshow(segmentation[20], cmap="Greys_r")
#     plt.colorbar()

#     plt.figure()
#     plt.hist(image.flatten(), bins=50, range=(np.amin(image), np.amax(image)), density=True)
#     plt.hist(segmentation.flatten(), bins=50, range=(np.amin(image), np.amax(image)), density=True)
#     # plt.colorbar()
#     plt.show()


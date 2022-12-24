
# ============================================================================
"""Image processing functions."""

import mindspore
from mindspore import Tensor


def rgb_to_yuv(image):
    """
    Converting images from RGB space to YUV space.

    Args:
        image (ndarray): Input image.

    Returns:
        Ndarray, converted image.
    """

    rgb_to_yuv_kernel = Tensor([
        [0.299, -0.14714119, 0.61497538],
        [0.587, -0.28886916, -0.51496512],
        [0.114, 0.43601035, -0.10001026]
    ], dtype=mindspore.float32)

    # Convert the pixel value range from -1-1 to 0-1.
    image = (image + 1.0) / 2.0

    yuv_img = mindspore.numpy.tensordot(
        image,
        rgb_to_yuv_kernel,
        axes=([image.ndim - 3], [0]))

    return yuv_img

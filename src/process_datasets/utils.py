
"""Image processing functions."""

import os
import numpy as np
import cv2


def compute_data_mean(data_folder):
    """
    Compute mean of R, G, B.

    Args:
        data_folder (str): Path of data.

    Returns:
        Ndarray, a list of channel means.

    Examples:
        >>> compute_data_mean('./dataset/train_photo')
    """

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    for img_file in image_files:
        path = os.path.join(data_folder, img_file)
        image = cv2.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[..., ::-1]  # Convert to BGR for training


def normalize_input(images):
    """
    Convert the pixel value range from 0-255 to 0-1.

    Args:
        images (ndarray): A batch of input images.

    Returns:
        Ndarray, normalized data.
    """

    return images / 127.5 - 1.0

import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    imgs = []
    for i in glob.glob(dir_name + "/*/*/*.jpg"):
        imgs.append(i)

    pixel_values = []

    for img in imgs:
        image = Image.open(img).convert("L")
        arr = np.asarray(image) / 255.0
        pixel_values.extend(arr.flatten())

    mean = np.mean(pixel_values)
    std = np.std(pixel_values)
    ############################################################################
    # Student code end
    ############################################################################
    return mean, std

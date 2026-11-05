"""Preprocessing that preserves the input rather than automatically 'fixing' it."""
import cv2
import numpy as np


def prepare_image(image, max_dimension=1600):
    """
    Return a consistent BGR image and useful grayscale representation.

    Resizing is optional and only reduces very large inputs for faster experiments.
    No histogram equalization or denoising is performed here because those changes
    could conceal the degradation that later stages are meant to measure.
    """
    if image is None or image.size == 0:
        raise ValueError("prepare_image received an empty image.")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.copy()
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError("Expected a grayscale, BGR, or BGRA image.")

    height, width = bgr.shape[:2]
    largest = max(height, width)
    if max_dimension and largest > max_dimension:
        scale = max_dimension / float(largest)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "bgr": bgr,
        "gray": gray,
        "height": gray.shape[0],
        "width": gray.shape[1],
    }

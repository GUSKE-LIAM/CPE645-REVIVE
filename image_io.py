"""Image input/output helpers for REVIVE."""
from pathlib import Path
import cv2
import numpy as np


def load_image(path, grayscale=False):
    """Load an image and return uint8 pixels; raises clear errors on bad paths."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}. Replace the placeholder path with your image."
        )

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    if image.dtype != np.uint8:
        image = to_uint8(image)
    return image


def save_image(path, image):
    """Save an image, creating parent directories if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None:
        raise ValueError("Cannot save an empty image.")
    success = cv2.imwrite(str(path), image)
    if not success:
        raise OSError(f"Could not write image to: {path}")
    return path


def to_uint8(image):
    """Convert numeric image data to uint8 without assuming a fixed input range."""
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Image is empty.")
    if array.dtype == np.uint8:
        return array.copy()

    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Image contains no finite pixel values.")

    low = float(array[finite].min())
    high = float(array[finite].max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)

    # Float images in [0, 1] are scaled conventionally; other ranges are min-max mapped.
    if low >= 0.0 and high <= 1.0:
        array = array * 255.0
    else:
        array = (array - low) * (255.0 / (high - low))
    return np.clip(array, 0, 255).astype(np.uint8)

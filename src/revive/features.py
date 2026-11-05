"""Hand-engineered patch-level features for degradation estimation."""
import cv2
import numpy as np


def _gray(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _patches(height, width, patch_size, stride):
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y2 = min(y + patch_size, height)
            x2 = min(x + patch_size, width)
            if y2 - y >= 8 and x2 - x >= 8:
                yield x, y, x2, y2


def extract_patch_features(image, patch_size=48, stride=32):
    """
    Calculate interpretable features for each overlapping image patch.

    Scores are heuristic and must be calibrated on the team's validation images.
    Returns a list of dictionaries with coordinates and raw feature values.
    """
    gray = _gray(image).astype(np.uint8)
    height, width = gray.shape
    results = []

    # A small median estimate provides a high-frequency residual for noise scoring.
    median = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - median.astype(np.float32)

    # Directional morphological responses help expose thin bright/dark line artifacts.
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_h)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    top_hat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_v)
    black_hat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
    line_response = np.maximum.reduce([top_hat, black_hat, top_hat_v, black_hat_v])

    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    for x1, y1, x2, y2 in _patches(height, width, patch_size, stride):
        patch = gray[y1:y2, x1:x2]
        patch_residual = residual[y1:y2, x1:x2]
        patch_lap = laplacian[y1:y2, x1:x2]
        patch_lines = line_response[y1:y2, x1:x2]

        # Robust noise estimate: median absolute deviation of high-frequency residual.
        residual_median = np.median(patch_residual)
        noise_sigma = float(
            np.median(np.abs(patch_residual - residual_median)) / 0.6745
        )

        # High Laplacian energy often indicates sharp detail; low energy can indicate blur.
        lap_variance = float(np.var(patch_lap))

        # Contrast range and local standard deviation are simple fading indicators.
        p05, p95 = np.percentile(patch, [5, 95])
        dynamic_range = float(p95 - p05)
        local_contrast = float(np.std(patch))

        # Fraction of pixels with a strong directional line response.
        scratch_fraction = float(np.mean(patch_lines > 22))

        results.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "noise_sigma": noise_sigma,
            "laplacian_variance": lap_variance,
            "dynamic_range": dynamic_range,
            "local_contrast": local_contrast,
            "scratch_fraction": scratch_fraction,
        })

    return results

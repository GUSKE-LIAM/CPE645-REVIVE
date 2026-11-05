"""Create controlled synthetic degradation and ground-truth masks.

These are experimental corruption models, not claims that every real old-photo
artifact has exactly this appearance.
"""
import cv2
import numpy as np


def add_gaussian_noise(image, sigma=12, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    return result, mask


def add_blur(image, kernel_size=5):
    kernel_size = max(3, int(kernel_size) | 1)
    result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    return result, mask


def add_fading(image, strength=0.35):
    """Reduce contrast and move intensities toward a pale gray value."""
    strength = float(np.clip(strength, 0.0, 1.0))
    gray_level = 220.0
    result = image.astype(np.float32) * (1.0 - strength) + gray_level * strength
    result = np.clip(result, 0, 255).astype(np.uint8)
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    return result, mask


def add_scratches(image, count=18, max_width=3, rng=None):
    """Draw random thin lines and return their approximate ground-truth mask."""
    rng = rng or np.random.default_rng()
    result = image.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(int(count)):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        # Mostly short-to-medium lines, with varied orientation.
        length = int(rng.integers(max(8, min(height, width) // 30),
                                  max(9, min(height, width) // 5)))
        angle = float(rng.uniform(0, np.pi))
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, width - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, height - 1))
        thickness = int(rng.integers(1, max(2, max_width + 1)))
        color_value = int(rng.choice([0, 255]))

        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        if result.ndim == 2:
            cv2.line(result, (x1, y1), (x2, y2), color_value, thickness)
        else:
            cv2.line(result, (x1, y1), (x2, y2),
                     (color_value, color_value, color_value), thickness)

    return result, mask


def create_damaged_image(image, damage_types=None, seed=42):
    """
    Apply selected corruptions in sequence.

    Returns:
        damaged: corrupted image
        masks: dictionary of approximate affected-pixel masks
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    damage_types = damage_types or ["noise", "blur", "fading", "scratches"]
    rng = np.random.default_rng(seed)
    damaged = image.copy()
    masks = {}

    allowed = {"noise", "blur", "fading", "scratches"}
    unknown = set(damage_types) - allowed
    if unknown:
        raise ValueError(f"Unknown damage types: {sorted(unknown)}")

    for damage in damage_types:
        if damage == "noise":
            damaged, masks[damage] = add_gaussian_noise(damaged, sigma=12, rng=rng)
        elif damage == "blur":
            damaged, masks[damage] = add_blur(damaged, kernel_size=5)
        elif damage == "fading":
            damaged, masks[damage] = add_fading(damaged, strength=0.35)
        elif damage == "scratches":
            damaged, masks[damage] = add_scratches(damaged, count=18, rng=rng)

    return damaged, masks

"""Convert hand-engineered patch features into regional damage maps."""
import numpy as np
from .features import extract_patch_features


DEFAULT_THRESHOLDS = {
    "noise": 0.35,
    "blur": 0.45,
    "fading": 0.45,
    "scratches": 0.025,
}


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def score_features(feature):
    """
    Convert raw patch measurements into heuristic 0–1 damage scores.

    These mappings are intentionally transparent starting points. Tune them
    against synthetic data rather than treating them as universal constants.
    """
    noise_score = _clip01(feature["noise_sigma"] / 24.0)

    # A simple inverse mapping; highly textured images can also have high Laplacian
    # variance, so blur scoring should be validated on varied image content.
    blur_score = _clip01(1.0 - feature["laplacian_variance"] / 450.0)

    # Low percentile dynamic range and low standard deviation indicate low contrast.
    fading_from_range = 1.0 - _clip01(feature["dynamic_range"] / 180.0)
    fading_from_std = 1.0 - _clip01(feature["local_contrast"] / 65.0)
    fading_score = _clip01(0.6 * fading_from_range + 0.4 * fading_from_std)

    scratch_score = _clip01(feature["scratch_fraction"] / 0.12)

    return {
        "noise": noise_score,
        "blur": blur_score,
        "fading": fading_score,
        "scratches": scratch_score,
    }


def build_damage_maps(image, patch_size=48, stride=32, thresholds=None):
    """
    Build full-resolution score maps and binary detection masks.

    Overlapping patch scores are accumulated and averaged over covered pixels
    to avoid hard rectangular patch boundaries.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    features = extract_patch_features(image, patch_size, stride)
    height, width = image.shape[:2]
    damage_types = ["noise", "blur", "fading", "scratches"]

    sums = {name: np.zeros((height, width), dtype=np.float32) for name in damage_types}
    counts = np.zeros((height, width), dtype=np.float32)

    scored_patches = []
    for feature in features:
        scores = score_features(feature)
        x1, y1, x2, y2 = (feature[k] for k in ("x1", "y1", "x2", "y2"))
        counts[y1:y2, x1:x2] += 1.0
        for name in damage_types:
            sums[name][y1:y2, x1:x2] += scores[name]
        scored_patches.append({**feature, **{f"{k}_score": v for k, v in scores.items()}})

    denominator = np.maximum(counts, 1.0)
    score_maps = {name: sums[name] / denominator for name in damage_types}
    detection_masks = {
        name: (score_maps[name] >= thresholds[name]).astype(np.uint8) * 255
        for name in damage_types
    }

    return {
        "features": scored_patches,
        "score_maps": score_maps,
        "detection_masks": detection_masks,
        "thresholds": thresholds,
    }

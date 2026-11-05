"""Aggregate regional damage scores into severity maps."""
import numpy as np


DEFAULT_WEIGHTS = {
    "noise": 0.25,
    "blur": 0.25,
    "fading": 0.20,
    "scratches": 0.30,
}


def calculate_severity(score_maps, weights=None):
    """
    Combine individual damage maps into an overall 0–1 severity map.

    Weights are explicit research parameters. Adjust and justify them during
    experiments; do not interpret this as a learned or clinically calibrated score.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_weight = sum(max(0.0, float(v)) for v in weights.values())
    if total_weight <= 0:
        raise ValueError("At least one positive severity weight is required.")

    first = next(iter(score_maps.values()))
    severity = np.zeros(first.shape, dtype=np.float32)

    for damage_type, weight in weights.items():
        if damage_type not in score_maps:
            continue
        severity += np.clip(score_maps[damage_type], 0.0, 1.0) * max(0.0, weight)

    severity /= total_weight
    return np.clip(severity, 0.0, 1.0)


def severity_labels(severity_map):
    """Return a simple three-level severity image: 0=low, 1=moderate, 2=high."""
    labels = np.zeros(severity_map.shape, dtype=np.uint8)
    labels[severity_map >= 0.35] = 1
    labels[severity_map >= 0.70] = 2
    return labels

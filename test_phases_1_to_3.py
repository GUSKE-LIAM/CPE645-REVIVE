"""Basic smoke tests for REVIVE phases 1–3."""
import numpy as np
from src.revive.preprocessing import prepare_image
from src.revive.damage.detection import build_damage_maps
from src.revive.damage.severity import calculate_severity
from src.revive.utils.synthetic_damage import create_damaged_image


def test_preprocessing_shapes():
    image = np.full((128, 160, 3), 120, dtype=np.uint8)
    result = prepare_image(image)
    assert result["bgr"].shape == image.shape
    assert result["gray"].shape == image.shape[:2]


def test_synthetic_damage_returns_masks():
    image = np.full((128, 160, 3), 120, dtype=np.uint8)
    damaged, masks = create_damaged_image(image, ["noise", "scratches"], seed=7)
    assert damaged.shape == image.shape
    assert set(masks) == {"noise", "scratches"}


def test_damage_maps_and_severity_shapes():
    image = np.random.default_rng(2).integers(0, 256, (128, 160, 3), dtype=np.uint8)
    analysis = build_damage_maps(image, patch_size=48, stride=32)
    severity = calculate_severity(analysis["score_maps"])
    assert severity.shape == image.shape[:2]
    assert np.isfinite(severity).all()
    assert severity.min() >= 0 and severity.max() <= 1

# REVIVE — Phases 1–3

This starter package implements:
1. Image loading, validation, and preprocessing.
2. Synthetic degradation generation with ground-truth masks.
3. Hand-engineered, patch-level damage features, detection maps, and severity maps.

The data folders are intentionally placeholders. Add your own permitted photographs under `data/`.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
```

Run the demo:

```bash
python -m src.revive.main --input data/original/REPLACE_WITH_IMAGE.jpg
```

The demo writes diagnostic maps to `outputs/`.

## Important scope note

The project uses OpenCV and NumPy as low-level computational building blocks (convolution, morphology, image I/O, and array operations). The feature definitions, patch scoring, thresholding, synthetic degradation generation, and severity aggregation are implemented in this repository. No pretrained restoration or damage-classification model is used.

The initial thresholds are heuristic starting points, not validated universal thresholds. Calibrate them using your synthetic image pairs and report their limitations.

## Data placeholders

- `data/original/`: photographs you want to analyze.
- `data/clean/`: clean source images used to generate synthetic pairs.
- `data/synthetic/`: optional saved synthetic damaged images.
- `data/real_damaged/`: real damaged photographs.

Do not commit private family photographs or large datasets without permission. The `.gitignore` excludes data contents by default.

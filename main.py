"""Small command-line demonstration of REVIVE phases 1–3."""
import argparse
from pathlib import Path
import cv2
import numpy as np

from .utils.image_io import load_image, save_image
from .preprocessing import prepare_image
from .damage.detection import build_damage_maps
from .damage.severity import calculate_severity, severity_labels


def main():
    parser = argparse.ArgumentParser(description="REVIVE damage analysis demo")
    parser.add_argument(
        "--input",
        default="data/original/REPLACE_WITH_IMAGE.jpg",
        help="Path to an input photograph."
    )
    parser.add_argument("--output", default="outputs", help="Output directory.")
    args = parser.parse_args()

    image = load_image(args.input)
    prepared = prepare_image(image)
    bgr = prepared["bgr"]

    analysis = build_damage_maps(bgr)
    severity = calculate_severity(analysis["score_maps"])
    labels = severity_labels(severity)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(output_dir / "input_prepared.png", bgr)
    save_image(output_dir / "overall_severity.png",
               np.clip(severity * 255, 0, 255).astype(np.uint8))
    save_image(output_dir / "severity_labels.png", labels * 127)

    for name, score_map in analysis["score_maps"].items():
        save_image(output_dir / f"{name}_score.png",
                   np.clip(score_map * 255, 0, 255).astype(np.uint8))
        save_image(output_dir / f"{name}_mask.png",
                   analysis["detection_masks"][name])

    print(f"Analyzed image: {args.input}")
    print(f"Patch count: {len(analysis['features'])}")
    print(f"Results saved to: {output_dir.resolve()}")
    print("Note: thresholds and score mappings are initial heuristics; validate them.")


if __name__ == "__main__":
    main()

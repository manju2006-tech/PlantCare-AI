import os
import sys
import json
import csv
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.unknown_detector import (
    UnknownPlantDetector,
    _build_extractor,
    _preprocess_image,
)

BASE = "trained_model_57class_run2"
MODEL_PATH = f"{BASE}/best_model.keras"
PROFILE_PATH = f"{BASE}/unknown_profile.npz"

UNKNOWN_DIR = "dataset/unsupported_plant_test/images"
METADATA_PATH = "dataset/unsupported_plant_test/metadata.csv"

MIN_CONFIDENCE = 50.0
BATCH_SIZE = 32

MULTIPLIERS = [
    0.70, 0.80, 0.90, 0.95, 1.00,
    1.05, 1.10, 1.20, 1.30, 1.50
]


def collect_images(folder):
    paths = []

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                paths.append(os.path.join(root, filename))

    return sorted(paths)


def main():
    print("Loading model and profile...")

    model = tf.keras.models.load_model(MODEL_PATH)

    detector = UnknownPlantDetector(PROFILE_PATH)
    detector.load(model)

    extractor = _build_extractor(model)

    centroids = detector.profile["centroids"]
    thresholds = detector.profile["thresholds"]
    classes = detector.class_names

    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        metadata = {
            row["filename"]: row["scientific_name"]
            for row in csv.DictReader(f)
        }

    paths = collect_images(UNKNOWN_DIR)

    if not paths:
        raise ValueError("No unsupported images found.")

    print(f"Unsupported images found: {len(paths)}")

    results = []

    for start in range(0, len(paths), BATCH_SIZE):
        chunk = paths[start:start + BATCH_SIZE]

        valid_paths = []
        images = []

        for path in chunk:
            try:
                images.append(_preprocess_image(path)[0])
                valid_paths.append(path)
            except Exception as exc:
                print(f"Skipping {path}: {exc}")

        if not images:
            continue

        batch = np.asarray(images, dtype=np.float32)

        probabilities = model.predict(batch, verbose=0)
        embeddings, _ = extractor.predict(batch, verbose=0)

        embeddings = np.asarray(embeddings).reshape(
            len(batch), -1
        )

        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :],
            axis=2
        )

        ratios = distances / np.maximum(
            thresholds[None, :], 1e-8
        )

        for i, path in enumerate(valid_paths):
            predicted_index = int(np.argmax(probabilities[i]))
            confidence = float(
                probabilities[i, predicted_index] * 100
            )

            nearest_index = int(np.argmin(ratios[i]))

            predicted_ratio = float(
                ratios[i, predicted_index]
            )

            nearest_ratio = float(
                ratios[i, nearest_index]
            )

            agreement = predicted_index == nearest_index

            results.append({
                "filename": os.path.basename(path),
                "category": metadata.get(
                    os.path.basename(path),
                    "unknown_category"
                ),
                "confidence": confidence,
                "predicted_ratio": predicted_ratio,
                "nearest_ratio": nearest_ratio,
                "agreement": agreement,
            })

        print(
            f"Processed {min(start + BATCH_SIZE, len(paths))}"
            f"/{len(paths)}",
            flush=True
        )

    if not results:
        raise ValueError("No readable unsupported images evaluated.")

    total = len(results)

    print("\nUNSUPPORTED PLANT MULTIPLIER COMPARISON")
    print("=" * 85)

    print(
        f"{'Multiplier':<13}"
        f"{'A Rejected':<18}"
        f"{'B Rejected':<18}"
        f"{'C Rejected':<18}"
    )

    print("-" * 67)

    for multiplier in MULTIPLIERS:
        a_rejected = sum(
            not (
                r["confidence"] >= MIN_CONFIDENCE
                and r["predicted_ratio"] <= multiplier
            )
            for r in results
        )

        b_rejected = sum(
            not (
                r["confidence"] >= MIN_CONFIDENCE
                and r["nearest_ratio"] <= multiplier
            )
            for r in results
        )

        c_rejected = sum(
            not (
                r["confidence"] >= MIN_CONFIDENCE
                and r["nearest_ratio"] <= multiplier
                and r["agreement"]
            )
            for r in results
        )

        print(
            f"{multiplier:<13.2f}"
            f"{100*a_rejected/total:<18.2f}"
            f"{100*b_rejected/total:<18.2f}"
            f"{100*c_rejected/total:<18.2f}"
        )

    print("\nCATEGORY RESULTS AT MULTIPLIER 1.0")
    print("-" * 67)

    for category in sorted(set(r["category"] for r in results)):
        subset = [
            r for r in results if r["category"] == category
        ]

        rejected = sum(
            not (
                r["confidence"] >= MIN_CONFIDENCE
                and r["nearest_ratio"] <= 1.0
                and r["agreement"]
            )
            for r in subset
        )

        print(
            f"{category}: "
            f"{rejected}/{len(subset)} rejected "
            f"({100*rejected/len(subset):.2f}%)"
        )

    print("\nExperiment complete.")
    print("No existing model, profile or application files modified.")


if __name__ == "__main__":
    main()
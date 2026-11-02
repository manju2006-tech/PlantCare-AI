
"""
PlantCare AI — Per-Class Threshold Analysis

Uses the existing 57-class model and unknown profile.
Analyses validation data without modifying existing files.
"""

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
    _build_extractor,
    _preprocess_image,
)

BASE = "trained_model_57class_run2"

MODEL_PATH = f"{BASE}/best_model.keras"
PROFILE_PATH = f"{BASE}/unknown_profile.npz"
CLASS_NAMES_PATH = f"{BASE}/class_names.json"

VALIDATION_DIR = "dataset/plantcare_57class/validation"

BATCH_SIZE = 32
MIN_CONFIDENCE = 50.0

TARGET_ACCEPTANCE = 95.0


def load_class_names(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    return [
        data[str(i)] if str(i) in data else data[i]
        for i in range(len(data))
    ]


def collect_images(folder, classes):
    samples = []

    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(folder, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing validation folder: {class_dir}"
            )

        for root, _, filenames in os.walk(class_dir):
            for filename in filenames:
                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    samples.append((
                        os.path.join(root, filename),
                        class_index,
                    ))

    return samples


def main():
    for path in (
        MODEL_PATH,
        PROFILE_PATH,
        CLASS_NAMES_PATH,
        VALIDATION_DIR,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing path: {path}")

    classes = load_class_names(CLASS_NAMES_PATH)

    with np.load(PROFILE_PATH, allow_pickle=False) as data:
        profile_classes = data["class_names"].astype(str).tolist()
        centroids = data["centroids"].astype(np.float32)
        thresholds = data["thresholds"].astype(np.float32)

    if classes != profile_classes:
        raise ValueError(
            "Model and profile class order does not match."
        )

    if len(classes) != 57:
        raise ValueError("Expected exactly 57 classes.")

    print("Loading existing model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    extractor = _build_extractor(model)

    samples = collect_images(VALIDATION_DIR, classes)

    print(f"Validation images found: {len(samples)}")
    print("Calculating class-wise distance ratios...\n")

    ratios_by_class = {
        name: [] for name in classes
    }

    correct = 0
    evaluated = 0
    skipped = 0

    for start in range(0, len(samples), BATCH_SIZE):
        chunk = samples[start:start + BATCH_SIZE]

        images = []
        valid_samples = []

        for path, actual_index in chunk:
            try:
                images.append(_preprocess_image(path)[0])
                valid_samples.append((path, actual_index))
            except Exception as exc:
                print(f"Skipping unreadable image: {path}: {exc}")
                skipped += 1

        if not images:
            continue

        batch = np.asarray(images, dtype=np.float32)

        probabilities = model.predict(batch, verbose=0)
        embeddings, _ = extractor.predict(batch, verbose=0)

        embeddings = np.asarray(embeddings).reshape(
            len(batch), -1
        )

        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1) * 100.0

        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :],
            axis=2,
        )

        for i, (_, actual_index) in enumerate(valid_samples):
            predicted_index = int(predictions[i])
            confidence = float(confidences[i])

            evaluated += 1

            if predicted_index == actual_index:
                correct += 1

            if confidence < MIN_CONFIDENCE:
                continue

            predicted_class = classes[predicted_index]

            ratio = (
                distances[i, predicted_index]
                / max(float(thresholds[predicted_index]), 1e-8)
            )

            ratios_by_class[predicted_class].append(
                float(ratio)
            )

        print(
            f"Processed {min(start + BATCH_SIZE, len(samples))}"
            f"/{len(samples)}",
            end="\r",
            flush=True,
        )

    print("\n\nCLASS-WISE CALIBRATION RESULTS")
    print("=" * 105)

    print(
        f"{'Class':<55}"
        f"{'Samples':<10}"
        f"{'Current':<12}"
        f"{'Proposed':<12}"
        f"{'P90':<10}"
        f"{'P95':<10}"
        f"{'P99':<10}"
    )

    print("-" * 105)

    proposed_multipliers = {}
    class_results = {}

    for class_name in classes:
        values = np.asarray(
            ratios_by_class[class_name],
            dtype=np.float32,
        )

        if len(values) == 0:
            print(f"{class_name:<55} No eligible samples")
            continue

        p90, p95, p99 = np.percentile(
            values, [90, 95, 99]
        )

        current_acceptance = (
            np.mean(values <= 1.0) * 100.0
        )

        proposed = float(p95)

        proposed_multipliers[class_name] = proposed

        proposed_acceptance = (
            np.mean(values <= proposed) * 100.0
        )

        class_results[class_name] = {
            "samples": int(len(values)),
            "current_acceptance": float(current_acceptance),
            "proposed_multiplier": proposed,
            "proposed_acceptance": float(proposed_acceptance),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
        }

        print(
            f"{class_name:<55}"
            f"{len(values):<10}"
            f"{current_acceptance:<12.2f}"
            f"{proposed:<12.4f}"
            f"{p90:<10.4f}"
            f"{p95:<10.4f}"
            f"{p99:<10.4f}"
        )

    if not class_results:
        raise ValueError("No eligible validation predictions found.")

    all_ratios = [
        value
        for values in ratios_by_class.values()
        for value in values
    ]

    print("\nOVERALL RESULTS")
    print("-" * 60)

    print(f"Images evaluated: {evaluated}")
    print(f"Unreadable images skipped: {skipped}")
    print(
        f"Model accuracy: "
        f"{100 * correct / evaluated:.2f}%"
    )
    print(f"Images with confidence >= {MIN_CONFIDENCE}%: {len(all_ratios)}")

    print(
        f"Current Rule A acceptance: "
        f"{100 * np.mean(np.asarray(all_ratios) <= 1.0):.2f}%"
    )

    calibrated_acceptance = []

    for class_name, values in ratios_by_class.items():
        if not values:
            continue

        calibrated_acceptance.extend(
            np.asarray(values) <= proposed_multipliers[class_name]
        )

    print(
        f"Proposed per-class acceptance: "
        f"{100 * np.mean(calibrated_acceptance):.2f}%"
    )

    output_path = "class_threshold_analysis.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "class_name",
            "samples",
            "current_acceptance_percent",
            "proposed_multiplier",
            "proposed_acceptance_percent",
            "p90",
            "p95",
            "p99",
        ])

        for class_name, result in class_results.items():
            writer.writerow([
                class_name,
                result["samples"],
                result["current_acceptance"],
                result["proposed_multiplier"],
                result["proposed_acceptance"],
                result["p90"],
                result["p95"],
                result["p99"],
            ])

    print(f"\nResults saved to: {output_path}")
    print("Existing model and detector profile were not modified.")


if __name__ == "__main__":
    main()

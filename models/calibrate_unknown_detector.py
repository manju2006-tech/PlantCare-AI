"""
PlantCare AI — Unknown Detector Calibration Experiment

Uses the existing model, profile and supported validation dataset.
Does NOT modify any existing model or profile files.
"""

import argparse
import json
import os
import sys

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


MODEL_PATH = "trained_model_57class_run2/best_model.keras"
PROFILE_PATH = "trained_model_57class_run2/unknown_profile.npz"
CLASS_NAMES_PATH = "trained_model_57class_run2/class_names.json"
VALIDATION_DIR = "dataset/plantcare_57class/validation"

MIN_CONFIDENCE = 50.0
BATCH_SIZE = 32

MULTIPLIERS = [
    0.70, 0.80, 0.90, 0.95, 1.00,
    1.05, 1.10, 1.20, 1.30, 1.50,
]


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    return [
        data[str(i)] if str(i) in data else data[i]
        for i in range(len(data))
    ]


def load_profile(path):
    with np.load(path, allow_pickle=False) as data:
        names = data["class_names"].astype(str).tolist()
        centroids = data["centroids"].astype(np.float32)
        thresholds = data["thresholds"].astype(np.float32)

    if len(names) != len(centroids):
        raise ValueError("Profile class and centroid counts differ.")

    if len(names) != len(thresholds):
        raise ValueError("Profile class and threshold counts differ.")

    if np.any(thresholds <= 0):
        raise ValueError("Profile contains non-positive thresholds.")

    return names, centroids, thresholds


def collect_validation_images(dataset_dir, class_names):
    samples = []

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Missing validation class folder: {class_dir}"
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


def evaluate(model, extractor, samples, centroids, thresholds):
    all_labels = []
    all_predictions = []
    all_confidences = []
    all_distances = []

    for start in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[start:start + BATCH_SIZE]
        images = []
        labels = []

        for image_path, label in batch_samples:
            try:
                image = _preprocess_image(image_path)[0]
                images.append(image)
                labels.append(label)
            except Exception as exc:
                print(f"Skipping unreadable image: {image_path}")
                print(f"Reason: {exc}")

        if not images:
            continue

        batch = np.asarray(images, dtype=np.float32)

        predictions = model.predict(batch, verbose=0)
        embeddings, _ = extractor.predict(batch, verbose=0)

        embeddings = np.asarray(embeddings).reshape(
            len(batch), -1
        )

        predictions = np.asarray(predictions)

        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1) * 100.0

        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :],
            axis=2,
        )

        all_labels.extend(labels)
        all_predictions.extend(predicted_classes)
        all_confidences.extend(confidences)
        all_distances.extend(distances)

        print(
            f"Processed {min(start + BATCH_SIZE, len(samples))}"
            f"/{len(samples)} images",
            end="\r",
            flush=True,
        )

    print()

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_confidences),
        np.asarray(all_distances),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation_dir",
        default=VALIDATION_DIR,
    )
    parser.add_argument(
        "--model_path",
        default=MODEL_PATH,
    )
    parser.add_argument(
        "--profile_path",
        default=PROFILE_PATH,
    )
    parser.add_argument(
        "--class_names",
        default=CLASS_NAMES_PATH,
    )
    args = parser.parse_args()

    for path in (
        args.validation_dir,
        args.model_path,
        args.profile_path,
        args.class_names,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    class_names = load_class_names(args.class_names)
    profile_names, centroids, thresholds = load_profile(
        args.profile_path
    )

    if class_names != profile_names:
        raise ValueError(
            "Model class order and profile class order do not match."
        )

    if len(class_names) != 57:
        raise ValueError(
            f"Expected 57 classes, found {len(class_names)}."
        )

    print("Loading existing 57-class model...")
    model = tf.keras.models.load_model(args.model_path)

    if model.output_shape[-1] != len(class_names):
        raise ValueError("Model output count does not match class names.")

    print("Building feature extractor...")
    extractor = _build_extractor(model)

    print("Collecting validation images...")
    samples = collect_validation_images(
        args.validation_dir,
        class_names,
    )

    if not samples:
        raise ValueError("No validation images were found.")

    print(f"Validation images found: {len(samples)}")
    print("Starting calibration experiment...\n")

    labels, predictions, confidences, distances = evaluate(
        model,
        extractor,
        samples,
        centroids,
        thresholds,
    )

    if len(labels) == 0:
        raise ValueError("No readable validation images were evaluated.")

    # Distance divided by the corresponding class threshold.
    normalized_distances = distances / thresholds[None, :]

    rows = np.arange(len(labels))

    predicted_ratios = normalized_distances[
        rows, predictions
    ]

    nearest_indices = np.argmin(normalized_distances, axis=1)
    nearest_ratios = normalized_distances[
        rows, nearest_indices
    ]

    confidence_ok = confidences >= MIN_CONFIDENCE
    nearest_matches_prediction = nearest_indices == predictions

    accuracy = np.mean(predictions == labels) * 100.0

    print("=" * 75)
    print("SUPPORTED VALIDATION RESULTS")
    print("=" * 75)
    print(f"Evaluated images: {len(labels)}")
    print(f"Classification accuracy: {accuracy:.2f}%")
    print(
        f"Images meeting confidence >= {MIN_CONFIDENCE:.0f}%: "
        f"{np.mean(confidence_ok) * 100:.2f}%"
    )

    print("\nRULE COMPARISON")
    print(
        f"{'Multiplier':<13}"
        f"{'Rule A Accept':<18}"
        f"{'Rule B Accept':<18}"
        f"{'Rule C Accept':<18}"
    )

    print("-" * 67)

    for multiplier in MULTIPLIERS:
        rule_a = (
            confidence_ok
            & (predicted_ratios <= multiplier)
        )

        rule_b = (
            confidence_ok
            & (nearest_ratios <= multiplier)
        )

        rule_c = (
            rule_b
            & nearest_matches_prediction
        )

        a_rate = np.mean(rule_a) * 100.0
        b_rate = np.mean(rule_b) * 100.0
        c_rate = np.mean(rule_c) * 100.0

        print(
            f"{multiplier:<13.2f}"
            f"{a_rate:<18.2f}"
            f"{b_rate:<18.2f}"
            f"{c_rate:<18.2f}"
        )

    print("\nCLASS-WISE CURRENT RULE A ACCEPTANCE")
    print("-" * 55)

    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index

        if not np.any(mask):
            print(f"{class_name}: no validation images")
            continue

        accepted = (
            confidence_ok[mask]
            & (predicted_ratios[mask] <= 1.0)
        )

        print(
            f"{class_name}: "
            f"{np.sum(accepted)}/{np.sum(mask)} "
            f"({np.mean(accepted) * 100:.2f}%)"
        )

    print("\nExperiment completed.")
    print("No existing project files were modified.")


if __name__ == "__main__":
    main()

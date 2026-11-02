
import os
import sys
import json
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from utils.unknown_detector import (
    UnknownPlantDetector,
    _build_extractor,
    _preprocess_image,
)

BASE = os.path.join(PROJECT_ROOT, "trained_model_57class_run2")
MODEL_PATH = os.path.join(BASE, "best_model.keras")
PROFILE_PATH = os.path.join(BASE, "unknown_profile.npz")
CLASSES_PATH = os.path.join(BASE, "class_names.json")
TEST_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "plantcare_57class", "test"
)
UNKNOWN_DIR = os.path.join(PROJECT_ROOT, "dataset", "unknown_test")

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BATCH_SIZE = 32

# Test several confidence cutoffs and multipliers for the
# predicted class's existing distance threshold.
CONFIDENCE_CUTOFFS = (50, 70, 80, 90, 95, 99)
DISTANCE_MULTIPLIERS = (0.70, 0.80, 0.90, 1.00, 1.10)


def collect_images(folder):
    paths = []
    if not os.path.isdir(folder):
        return paths

    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(EXTENSIONS):
                paths.append(os.path.join(root, filename))

    return sorted(paths)


def load_batch(paths):
    valid_paths = []
    images = []

    for path in paths:
        try:
            images.append(_preprocess_image(path)[0])
            valid_paths.append(path)
        except Exception as exc:
            print(f"\nSkipped unreadable image {path}: {exc}")

    if not images:
        return [], None

    return valid_paths, np.asarray(images, dtype=np.float32)


def main():
    print("Loading model and detector...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    if isinstance(classes, dict):
        classes = [
            classes[str(i)] if str(i) in classes else classes[i]
            for i in range(len(classes))
        ]

    detector = UnknownPlantDetector(PROFILE_PATH)
    detector.load(model)
    extractor = _build_extractor(model)

    names = detector.class_names
    centroids = detector.profile["centroids"]
    thresholds = detector.profile["thresholds"]
    profile_index = {name: i for i, name in enumerate(names)}

    supported_paths = []
    for class_name in classes:
        supported_paths.extend(
            collect_images(os.path.join(TEST_DIR, class_name))
        )

    unknown_paths = collect_images(UNKNOWN_DIR)

    print(f"Supported images: {len(supported_paths)}")
    print(f"Unknown images: {len(unknown_paths)}")
    print("Extracting supported-image features...", flush=True)

    # Store only confidence and predicted-class distance ratio,
    # so all threshold combinations can be tested without
    # running the model repeatedly.
    supported_confidence = []
    supported_ratio = []
    supported_correct = []
    skipped = 0

    for start in range(0, len(supported_paths), BATCH_SIZE):
        chunk = supported_paths[start:start + BATCH_SIZE]
        paths, batch = load_batch(chunk)

        if batch is None:
            skipped += len(chunk)
            continue

        embeddings, probabilities = extractor.predict(batch, verbose=0)
        embeddings = np.asarray(embeddings).reshape(len(paths), -1)
        probabilities = np.asarray(probabilities)

        for path, embedding, probs in zip(
            paths, embeddings, probabilities
        ):
            pred_index = int(np.argmax(probs))
            predicted = classes[pred_index]
            confidence = float(probs[pred_index]) * 100.0
            actual = os.path.basename(os.path.dirname(path))

            if predicted in profile_index:
                idx = profile_index[predicted]
                distance = float(
                    np.linalg.norm(embedding - centroids[idx])
                )
                ratio = distance / max(float(thresholds[idx]), 1e-8)
            else:
                ratio = float("inf")

            supported_confidence.append(confidence)
            supported_ratio.append(ratio)
            supported_correct.append(predicted == actual)

        processed = len(supported_confidence)
        if processed and processed % 640 < BATCH_SIZE:
            print(
                f"Progress: {processed}/{len(supported_paths)}",
                flush=True,
            )

    print("\nExtracting unknown-image features...", flush=True)
    unknown_records = []

    for start in range(0, len(unknown_paths), BATCH_SIZE):
        chunk = unknown_paths[start:start + BATCH_SIZE]
        paths, batch = load_batch(chunk)

        if batch is None:
            continue

        embeddings, probabilities = extractor.predict(batch, verbose=0)
        embeddings = np.asarray(embeddings).reshape(len(paths), -1)
        probabilities = np.asarray(probabilities)

        for path, embedding, probs in zip(
            paths, embeddings, probabilities
        ):
            pred_index = int(np.argmax(probs))
            predicted = classes[pred_index]
            confidence = float(probs[pred_index]) * 100.0

            if predicted in profile_index:
                idx = profile_index[predicted]
                distance = float(
                    np.linalg.norm(embedding - centroids[idx])
                )
                ratio = distance / max(float(thresholds[idx]), 1e-8)
            else:
                ratio = float("inf")

            unknown_records.append({
                "filename": os.path.basename(path),
                "predicted": predicted,
                "confidence": confidence,
                "ratio": ratio,
            })

    conf = np.asarray(supported_confidence)
    ratio = np.asarray(supported_ratio)
    correct = np.asarray(supported_correct, dtype=bool)
    total = len(conf)

    print("\nSUPPORTED DATASET")
    print(f"Evaluated: {total}; unreadable/skipped: {skipped}")
    if total:
        print(f"Model accuracy: {100 * correct.mean():.2f}%")

    print("\nTHRESHOLD SWEEP")
    print("Confidence | Distance multiplier | Accepted | Rejected | Acceptance %")
    print("-" * 75)

    table = []
    for min_conf in CONFIDENCE_CUTOFFS:
        for multiplier in DISTANCE_MULTIPLIERS:
            accepted = (conf >= min_conf) & (ratio <= multiplier)
            count = int(accepted.sum())
            percent = 100.0 * count / total if total else 0.0
            table.append((min_conf, multiplier, count, percent))
            print(
                f"{min_conf:10.1f} | {multiplier:19.2f} | "
                f"{count:8d} | {total-count:8d} | {percent:11.2f}%"
            )

    print("\nUNKNOWN IMAGE RESULTS")
    for record in unknown_records:
        print(
            f"\n{record['filename']}"
            f"\n  Predicted: {record['predicted']}"
            f"\n  Confidence: {record['confidence']:.2f}%"
            f"\n  Distance ratio: {record['ratio']:.4f}"
        )

        for min_conf, multiplier in [
            (70, 0.80),
            (80, 0.90),
            (90, 0.90),
            (95, 0.80),
        ]:
            accepted = (
                record["confidence"] >= min_conf
                and record["ratio"] <= multiplier
            )
            print(
                f"  conf>={min_conf}, ratio<={multiplier:.2f}: "
                f"{'ACCEPT' if accepted else 'REJECT'}"
            )

    print("\nSweep complete. No app files were changed.", flush=True)


if __name__ == "__main__":
    main()

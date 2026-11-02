
import os
import sys
import json
import numpy as np
import tensorflow as tf
import csv

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
UNKNOWN_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "unsupported_plant_test", "images"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "unsupported_plant_test", "metadata.csv"
)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BATCH_SIZE = 32
PROGRESS_EVERY = 640


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
    
    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        metadata_rows = csv.DictReader(f)
        species_by_filename = {
            row["filename"]: row["scientific_name"]
            for row in metadata_rows
        }

    print(f"Supported images: {len(supported_paths)}")
    print(f"Unknown images: {len(unknown_paths)}")
    print("Evaluating supported images...", flush=True)

    evaluated = skipped = correct = 0
    accepted = {"A": 0, "B": 0, "C": 0}
    unknown_results = []

    def evaluate(paths, embeddings, probabilities):
        results = []

        distances = np.linalg.norm(
            embeddings[:, None, :] - centroids[None, :, :],
            axis=2,
        )
        ratios = distances / np.maximum(thresholds[None, :], 1e-8)

        for i, path in enumerate(paths):
            predicted_index = int(np.argmax(probabilities[i]))
            predicted = classes[predicted_index]
            confidence = float(probabilities[i, predicted_index]) * 100
            nearest_index = int(np.argmin(ratios[i]))
            nearest = names[nearest_index]
            nearest_distance = float(distances[i, nearest_index])
            nearest_threshold = float(thresholds[nearest_index])
            nearest_ratio = float(ratios[i, nearest_index])

            # Rule A: current predicted-class centroid rule.
            if predicted in profile_index:
                pi = profile_index[predicted]
                predicted_distance = float(distances[i, pi])
                predicted_threshold = float(thresholds[pi])
                rule_a = (
                    confidence >= detector.min_confidence
                    and predicted_distance <= predicted_threshold
                )
            else:
                predicted_distance = None
                predicted_threshold = None
                rule_a = False

            # Rule B: nearest normalized distance.
            rule_b = (
                confidence >= detector.min_confidence
                and nearest_ratio <= 1.0
            )

            # Rule C: Rule B plus predicted/nearest class agreement.
            rule_c = rule_b and predicted == nearest
            
            
            # Experimental Rule D: current Rule A plus
            # a stricter confidence cutoff.
            rule_d_60 = rule_a and confidence >= 60.0
            rule_d_70 = rule_a and confidence >= 70.0
            rule_d_80 = rule_a and confidence >= 80.0


            results.append({
                "path": path,
                "predicted": predicted,
                "confidence": confidence,
                "nearest": nearest,
                "nearest_distance": nearest_distance,
                "nearest_threshold": nearest_threshold,
                "nearest_ratio": nearest_ratio,
                "predicted_distance": predicted_distance,
                "predicted_threshold": predicted_threshold,
                "A": rule_a,
                "B": rule_b,
                "C": rule_c,
                "D60": rule_d_60,
                "D70": rule_d_70,
                "D80": rule_d_80,
            })

        return results

    for start in range(0, len(supported_paths), BATCH_SIZE):
        chunk = supported_paths[start:start + BATCH_SIZE]
        paths, batch = load_batch(chunk)

        if batch is None:
            skipped += len(chunk)
            continue

        embeddings, probabilities = extractor.predict(batch, verbose=0)
        embeddings = np.asarray(embeddings).reshape(len(paths), -1)

        results = evaluate(paths, embeddings, np.asarray(probabilities))

        for r in results:
            evaluated += 1
            actual = os.path.basename(os.path.dirname(r["path"]))
            correct += int(r["predicted"] == actual)
            for rule in accepted:
                accepted[rule] += int(r[rule])

        if evaluated and evaluated % PROGRESS_EVERY < BATCH_SIZE:
            print(f"Progress: {evaluated}/{len(supported_paths)}", flush=True)

    print("\nSUPPORTED IMAGE COMPARISON")
    print(f"Evaluated: {evaluated}; skipped: {skipped}")
    if evaluated:
        print(f"Model classification accuracy: {100 * correct / evaluated:.2f}%")
        for rule, label in [
            ("A", "Current predicted-class rule"),
            ("B", "Nearest normalized-distance rule"),
            ("C", "Nearest distance + class agreement"),
        ]:
            n = accepted[rule]
            print(
                f"Rule {rule} ({label}): accepted {n} "
                f"({100 * n / evaluated:.2f}%), rejected {evaluated - n}"
            )

    print("\nEvaluating unknown images...", flush=True)

    for start in range(0, len(unknown_paths), BATCH_SIZE):
        chunk = unknown_paths[start:start + BATCH_SIZE]
        paths, batch = load_batch(chunk)

        if batch is None:
            continue

        embeddings, probabilities = extractor.predict(batch, verbose=0)
        embeddings = np.asarray(embeddings).reshape(len(paths), -1)
        results = evaluate(paths, embeddings, np.asarray(probabilities))

        for r in results:
            unknown_results.append(r)
            print(
                f"\n{os.path.basename(r['path'])}"
                f"\n  Predicted: {r['predicted']} ({r['confidence']:.2f}%)"
                f"\n  Nearest class: {r['nearest']}"
                f"\n  Nearest distance/threshold: "
                f"{r['nearest_distance']:.4f}/{r['nearest_threshold']:.4f}"
                f"\n  Normalized ratio: {r['nearest_ratio']:.4f}"
                f"\n  Rule A: {'ACCEPT' if r['A'] else 'REJECT'}"
                f"\n  Rule B: {'ACCEPT' if r['B'] else 'REJECT'}"
                f"\n  Rule C: {'ACCEPT' if r['C'] else 'REJECT'}"
            )

    
    print("\nUNKNOWN IMAGE COMPARISON")
    print(f"Evaluated: {len(unknown_results)}")

    total = len(unknown_results)
    if total:
        for rule, label in [
            ("A", "Current predicted-class rule"),
            ("B", "Nearest normalized-distance rule"),
            ("C", "Nearest distance + class agreement"),
        ]:
            n = sum(bool(r[rule]) for r in unknown_results)
            rejected = total - n
            print(
                f"Rule {rule} ({label}): "
                f"accepted {n} ({100 * n / total:.2f}%), "
                f"rejected {rejected} "
                f"({100 * rejected / total:.2f}%)"
            )

    
    print("\nUNKNOWN IMAGE RESULTS BY CATEGORY")

    category_results = {}

    for r in unknown_results:
        filename = os.path.basename(r["path"])
        category = species_by_filename.get(filename, "unknown_category")

        if category not in category_results:
            category_results[category] = {
                rule: {"accepted": 0, "rejected": 0}
                for rule in ("A", "B", "C")
            }

        for rule in ("A", "B", "C"):
            outcome = "accepted" if r[rule] else "rejected"
            category_results[category][rule][outcome] += 1

    for category in sorted(category_results):
        counts = category_results[category]
        total_category = counts["A"]["accepted"] + counts["A"]["rejected"]

        print(f"\n{category}: {total_category} images")

        for rule in ("A", "B", "C"):
            accepted_count = counts[rule]["accepted"]
            rejected_count = counts[rule]["rejected"]

            print(
                f"  Rule {rule}: "
                f"accepted {accepted_count} "
                f"({100 * accepted_count / total_category:.2f}%), "
                f"rejected {rejected_count} "
                f"({100 * rejected_count / total_category:.2f}%)"
            )
            
            
    print("\nEXPERIMENTAL CONFIDENCE CUTOFFS")

    for rule, cutoff in [
        ("D60", 60),
        ("D70", 70),
        ("D80", 80),
    ]:
        supported_accepted = 0
        supported_total = 0

        # Re-evaluate supported images separately in the next experiment.
        unknown_accepted = sum(bool(r[rule]) for r in unknown_results)
        unknown_total = len(unknown_results)

        if unknown_total:
            print(
                f"{rule} (confidence >= {cutoff}%): "
                f"unknown accepted {unknown_accepted}/{unknown_total} "
                f"({100 * unknown_accepted / unknown_total:.2f}%), "
                f"rejected {unknown_total - unknown_accepted}"
            )


    print("\nComparison complete. No app files were changed.", flush=True)




if __name__ == "__main__":
    main()

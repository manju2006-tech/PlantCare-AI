"""Evaluate PlantCare AI's unknown-plant detector on held-out images."""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.unknown_detector import UnknownPlantDetector


MODEL_PATH = os.path.join(
    PROJECT_ROOT, "trained_model_57class_run2", "best_model.keras"
)
CLASS_NAMES_PATH = os.path.join(
    PROJECT_ROOT, "trained_model_57class_run2", "class_names.json"
)
PROFILE_PATH = os.path.join(
    PROJECT_ROOT, "trained_model_57class_run2", "unknown_profile.npz"
)
TEST_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "plantcare_57class", "test"
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    for path in (MODEL_PATH, CLASS_NAMES_PATH, PROFILE_PATH, TEST_DIR):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    with open(CLASS_NAMES_PATH, encoding="utf-8") as file:
        class_names = json.load(file)

    if isinstance(class_names, dict):
        class_names = [
            class_names[str(i)] if str(i) in class_names else class_names[i]
            for i in range(len(class_names))
        ]

    model = tf.keras.models.load_model(MODEL_PATH)
    detector = UnknownPlantDetector(PROFILE_PATH)
    detector.load(model)

    total = 0
    accepted = 0
    rejected = 0
    correct_predictions = 0
    class_stats = defaultdict(lambda: [0, 0, 0])
    rejected_examples = []

    print("Evaluating supported test images...")
    print("This may take a few minutes.\n")

    for true_class in class_names:
        class_dir = os.path.join(TEST_DIR, true_class)
        if not os.path.isdir(class_dir):
            print(f"WARNING: Missing test folder: {true_class}")
            continue

        image_paths = []
        for root, _, filenames in os.walk(class_dir):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, filename))

        for image_path in image_paths:
            try:
                with tf.device("/CPU:0"):
                    from utils import model_loader

                    # Use the same preprocessing and prediction logic as the app.
                    # Initialize the model loader's globals for this standalone run.
                    model_loader._model = model
                    model_loader._index_to_class = {
                        str(i): name for i, name in enumerate(class_names)
                    }
                    model_loader._image_size = (224, 224)

                    prediction = model_loader.predict_image(image_path)
                    result = detector.check(image_path, prediction)

                total += 1
                stats = class_stats[true_class]
                stats[0] += 1

                if prediction["class_name"] == true_class:
                    correct_predictions += 1

                if result["is_known"]:
                    accepted += 1
                    stats[1] += 1
                else:
                    rejected += 1
                    if len(rejected_examples) < 20:
                        rejected_examples.append(
                            (
                                os.path.relpath(image_path, TEST_DIR),
                                result["reason"],
                                prediction["class_name"],
                                prediction["confidence"],
                            )
                        )

            except Exception as exc:
                print(f"ERROR processing {image_path}: {exc}")

    print("\n========== SUPPORTED TEST RESULTS ==========")
    print(f"Images evaluated: {total}")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(
        "Accepted rate: "
        f"{(accepted / total * 100) if total else 0:.2f}%"
    )
    print(
        "Model classification accuracy: "
        f"{(correct_predictions / total * 100) if total else 0:.2f}%"
    )

    print("\n========== PER-CLASS ACCEPTANCE ==========")
    for class_name in class_names:
        seen, kept, _ = class_stats[class_name]
        if seen:
            print(
                f"{class_name}: accepted {kept}/{seen} "
                f"({kept / seen * 100:.1f}%)"
            )

    if rejected_examples:
        print("\n========== FIRST REJECTED SUPPORTED IMAGES ==========")
        for path, reason, predicted, confidence in rejected_examples:
            print(
                f"{path} | reason={reason} | "
                f"predicted={predicted} | confidence={confidence:.2f}%"
            )

    print(
        "\nNote: this evaluates supported images only. "
        "It does not measure the wheat/unknown rejection rate."
    )


if __name__ == "__main__":
    main()

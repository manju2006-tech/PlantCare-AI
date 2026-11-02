import json
import os
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from utils import model_loader
from utils.unknown_detector import UnknownPlantDetector

MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model_57class_run2")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
PROFILE_PATH = os.path.join(MODEL_DIR, "unknown_profile.npz")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "dataset", "unknown_test")


def main():
    for path in (MODEL_PATH, PROFILE_PATH, CLASS_NAMES_PATH, IMAGE_DIR):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, encoding="utf-8") as file:
        classes = json.load(file)

    if isinstance(classes, dict):
        classes = [
            classes[str(i)] if str(i) in classes else classes[i]
            for i in range(len(classes))
        ]

    model_loader._model = model
    model_loader._index_to_class = {
        str(i): name for i, name in enumerate(classes)
    }
    model_loader._image_size = (224, 224)

    detector = UnknownPlantDetector(PROFILE_PATH)
    detector.load(model)

    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for filename in sorted(os.listdir(IMAGE_DIR)):
        if not filename.lower().endswith(extensions):
            continue

        path = os.path.join(IMAGE_DIR, filename)

        try:
            prediction = model_loader.predict_image(path)
            check = detector.check(path, prediction)

            print(f"\nImage: {filename}")
            print(f"Predicted class: {prediction['class_name']}")
            print(f"Model confidence: {prediction['confidence']:.2f}%")
            print(f"Detector decision: {'ACCEPT' if check['is_known'] else 'REJECT'}")
            print(f"Reason: {check['reason']}")

            if "distance" in check:
                print(f"Predicted-class distance: {check['distance']:.4f}")
                print(f"Predicted-class threshold: {check['threshold']:.4f}")

            if "nearest_class" in check:
                print(f"Nearest class: {check['nearest_class']}")
                print(f"Nearest-centroid distance: {check['nearest_distance']:.4f}")

        except Exception as exc:
            print(f"\nERROR processing {filename}: {exc}")


if __name__ == "__main__":
    main()
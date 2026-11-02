"""
Build the PlantCare AI unknown-plant reference profile.

Run from the project root:
python models/build_unknown_detector.py
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.unknown_detector import build_class_profile, save_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def load_class_names(path):
    """Support both a list and an index-to-class dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    return [
        data[str(i)] if str(i) in data else data[i]
        for i in range(len(data))
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Build an unknown-plant detection profile."
    )
    parser.add_argument(
        "--dataset_dir",
        default="dataset/plantcare_57class/train",
        help="Training directory containing one folder per class.",
    )
    parser.add_argument(
        "--model_path",
        default="trained_model_57class_run2/best_model.keras",
        help="Path to the existing trained Keras model.",
    )
    parser.add_argument(
        "--class_indices",
        default="trained_model_57class_run2/class_names.json",
        help="JSON file containing model class names.",
    )
    parser.add_argument(
        "--out",
        default="trained_model_57class_run2/unknown_profile.npz",
        help="Output profile path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Images processed per batch.",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    model_path = os.path.abspath(args.model_path)
    class_path = os.path.abspath(args.class_indices)
    output_path = os.path.abspath(args.out)

    for path in (dataset_dir, model_path, class_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    class_names = load_class_names(class_path)
    if len(class_names) != 57:
        raise ValueError(
            f"Expected 57 model classes, found {len(class_names)}."
        )

    dataset_classes = {
        name
        for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    }
    missing = sorted(set(class_names) - dataset_classes)

    if missing:
        raise ValueError(
            "The training directory is missing these model classes:\n"
            + "\n".join(missing)
        )

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model output has {model.output_shape[-1]} classes, "
            f"but class_names.json contains {len(class_names)}."
        )

    print(f"Building profiles from: {dataset_dir}")
    profile = build_class_profile(
        model=model,
        dataset_dir=dataset_dir,
        class_names=class_names,
        image_size=(224, 224),
        batch_size=args.batch_size,
    )

    save_profile(profile, output_path)

    print("\nProfile created successfully.")
    print(f"Classes profiled: {len(profile['class_names'])}/57")
    print(f"Saved to: {output_path}")

    if len(profile["class_names"]) != 57:
        print(
            "WARNING: Some classes were skipped. Do not enable the "
            "runtime detector until all 57 classes are profiled."
        )


if __name__ == "__main__":
    main()

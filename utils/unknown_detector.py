"""
PlantCare AI — Unknown Plant Detector

Uses feature embeddings from the existing trained model to help
identify images that do not resemble the supported training classes.

This is a safeguard, not a guarantee. Some unfamiliar plants may
still be accepted, and some valid images may be rejected.
"""

import json
import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

PROFILE_NAME = "unknown_profile.npz"
DEFAULT_MIN_CONFIDENCE = 50.0
DEFAULT_PERCENTILE = 5.0


def _load_class_names(path):
    """Load class names from a list or an index-to-class dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    return [
        data[str(i)] if str(i) in data else data[i]
        for i in range(len(data))
    ]


def _preprocess_image(image_path, image_size=(224, 224)):
    """Apply the same MobileNetV2 preprocessing as the existing app."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (image_size[1], image_size[0]),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32)

    array = array / 127.5 - 1.0
    return np.expand_dims(array, axis=0)


def _build_extractor(model):
    """
    Extract the input to the final Dense classification layer.
    The original model remains unchanged.
    """
    import tensorflow as tf

    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    if not dense_layers:
        raise ValueError(
            "No Dense classification layer was found in the model."
        )

    final_dense = dense_layers[-1]
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[final_dense.input, model.output],
    )


def build_class_profile(
    model,
    dataset_dir,
    class_names,
    image_size=(224, 224),
    batch_size=32,
):
    """
    Build a feature profile from dataset/<class_name> folders.

    Returns a dictionary containing class names, feature centroids,
    and per-class distance thresholds.
    """
    extractor = _build_extractor(model)
    centroids = []
    thresholds = []
    valid_names = []

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            logger.warning("Missing class folder: %s", class_dir)
            continue

        image_paths = []
        for root, _, filenames in os.walk(class_dir):
            for filename in filenames:
                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    image_paths.append(os.path.join(root, filename))

        if len(image_paths) < 8:
            logger.warning(
                "Skipping %s: only %d images found (need at least 8).",
                class_name,
                len(image_paths),
            )
            continue

        features = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    batch_images.append(
                        _preprocess_image(path, image_size)[0]
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not read image %s: %s", path, exc
                    )

            if not batch_images:
                continue

            batch = np.asarray(batch_images, dtype=np.float32)
            embeddings, _ = extractor.predict(batch, verbose=0)
            embeddings = np.asarray(embeddings).reshape(
                len(batch), -1
            )
            features.extend(embeddings)

        if len(features) < 8:
            logger.warning(
                "Skipping %s: too few readable images.", class_name
            )
            continue

        features = np.asarray(features, dtype=np.float32)
        centroid = features.mean(axis=0)

        # Euclidean distances from training images to their class centre.
        distances = np.linalg.norm(features - centroid, axis=1)

        # A conservative initial threshold; calibration may be needed.
        threshold = float(np.percentile(distances, 95))

        centroids.append(centroid)
        thresholds.append(max(threshold, 1e-8))
        valid_names.append(class_name)

        logger.info(
            "Profiled %s using %d images.", class_name, len(features)
        )

    if not valid_names:
        raise ValueError(
            "No class profiles could be built. Check the dataset path."
        )

    return {
        "class_names": valid_names,
        "centroids": np.asarray(centroids, dtype=np.float32),
        "thresholds": np.asarray(thresholds, dtype=np.float32),
    }


def save_profile(profile, output_path):
    """Save the detector profile as a NumPy archive."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    np.savez_compressed(
        output_path,
        class_names=np.asarray(profile["class_names"]),
        centroids=profile["centroids"],
        thresholds=profile["thresholds"],
    )


class UnknownPlantDetector:
    """Runtime detector backed by a saved feature profile."""

    def __init__(self, profile_path, min_confidence=DEFAULT_MIN_CONFIDENCE):
        self.profile_path = profile_path
        self.min_confidence = float(min_confidence)
        self.profile = None
        self.extractor = None
        self.class_names = []

    def load(self, model):
        if not os.path.isfile(self.profile_path):
            raise FileNotFoundError(
                f"Detector profile not found: {self.profile_path}"
            )

        with np.load(self.profile_path, allow_pickle=False) as data:
            self.class_names = data["class_names"].astype(str).tolist()
            centroids = data["centroids"].astype(np.float32)
            thresholds = data["thresholds"].astype(np.float32)

        if len(self.class_names) != len(centroids):
            raise ValueError("Detector profile has inconsistent class data.")

        if len(centroids) != len(thresholds):
            raise ValueError("Detector profile has inconsistent thresholds.")

        self.profile = {
            "centroids": centroids,
            "thresholds": thresholds,
        }
        self.extractor = _build_extractor(model)

    def check(self, image_path, prediction, image_size=(224, 224)):
        """
        Return whether the image resembles a supported class.

        prediction should be the existing model_loader prediction result.
        """
        if self.profile is None or self.extractor is None:
            raise RuntimeError("Detector has not been loaded.")

        confidence = float(prediction.get("confidence", 0.0))
        predicted_name = prediction.get("class_name", "")

        if confidence < self.min_confidence:
            return {
                "is_known": False,
                "reason": "low_confidence",
                "confidence": confidence,
            }

        if predicted_name not in self.class_names:
            return {
                "is_known": False,
                "reason": "unsupported_class",
                "confidence": confidence,
            }

        image = _preprocess_image(image_path, image_size)
        embeddings, _ = self.extractor.predict(image, verbose=0)
        embedding = np.asarray(embeddings).reshape(1, -1)[0]

                # Compare the image against ALL supported class centroids
        centroids = self.profile["centroids"]
        thresholds = self.profile["thresholds"]

        distances = np.linalg.norm(
            centroids - embedding.reshape(1, -1),
            axis=1,
        )

        nearest_index = int(np.argmin(distances))
        nearest_class = self.class_names[nearest_index]
        nearest_distance = float(distances[nearest_index])
        nearest_threshold = float(thresholds[nearest_index])

        # Keep the original decision for now.
        class_index = self.class_names.index(predicted_name)
        centroid = centroids[class_index]
        threshold = float(thresholds[class_index])
        distance = float(np.linalg.norm(embedding - centroid))
        is_known = distance <= threshold

        return {
            "is_known": bool(is_known),
            "reason": "matched_profile" if is_known else "feature_distance",
            "confidence": confidence,
            "distance": distance,
            "threshold": float(threshold),
            "class_name": predicted_name,
            "nearest_class": nearest_class,
            "nearest_distance": nearest_distance,
        }

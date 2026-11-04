"""Learned supported/unsupported rejection using the frozen disease features.

The artifact contains only numeric arrays and JSON metadata (no pickle).
Its decision score is a classifier margin, not an identification probability.
"""
import hashlib
import json

import numpy as np
from PIL import Image

FORMAT_VERSION = 1
PREPROCESSING = "rgb_bicubic_mobilenet_v2"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_image(image_path, image_size=(224, 224)):
    """Same RGB resize and scaling used by disease inference and training."""
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(
            (image_size[1], image_size[0]), Image.Resampling.BICUBIC
        )
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array / 127.5 - 1.0, 0)


def normalize_features(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Expected a finite matrix of image features.")
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError("Image features have zero length.")
    return features / lengths


def kernel_scores(features, support_vectors, dual_coef, intercept, gamma):
    """Evaluate a binary RBF SVM without loading executable model objects."""
    features = normalize_features(features).astype(np.float64)
    squared_distances = np.maximum(
        np.sum(features ** 2, axis=1)[:, None]
        + np.sum(support_vectors ** 2, axis=1)[None, :]
        - 2.0 * features @ support_vectors.T,
        0.0,
    )
    return np.exp(-gamma * squared_distances) @ dual_coef + intercept


class SupportedPlantDetector:
    def __init__(self, artifact_path, min_confidence=50.0):
        self.artifact_path = artifact_path
        self.min_confidence = float(min_confidence)
        self.extractor = None

    def load(self, model, class_names, model_path):
        from utils.unknown_detector import _build_extractor

        self.extractor = None
        with np.load(self.artifact_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            support_vectors = data["support_vectors"].astype(np.float64)
            dual_coef = data["dual_coef"].astype(np.float64)
            intercept = float(data["intercept"].item())
            gamma = float(data["gamma"].item())
            threshold = float(data["threshold"].item())

        if metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported support-detector artifact version.")
        if metadata.get("preprocessing") != PREPROCESSING:
            raise ValueError("Support-detector preprocessing does not match.")
        if metadata.get("class_names") != list(class_names):
            raise ValueError("Support detector and disease labels do not match.")
        if metadata.get("disease_model_sha256") != file_sha256(model_path):
            raise ValueError("Support detector was trained for a different model.")
        image_size = tuple(metadata.get("image_size", ()))
        if len(image_size) != 2 or any(type(n) is not int or n <= 0 for n in image_size):
            raise ValueError("Invalid support-detector image size.")
        if support_vectors.ndim != 2 or not support_vectors.shape[0]:
            raise ValueError("Support detector has no reference vectors.")
        if dual_coef.shape != (len(support_vectors),):
            raise ValueError("Invalid support-detector coefficients.")
        if not all(np.isfinite(a).all() for a in (
            support_vectors, dual_coef, np.asarray([intercept, gamma, threshold])
        )) or gamma <= 0:
            raise ValueError("Invalid support-detector numeric data.")
        if not np.allclose(np.linalg.norm(support_vectors, axis=1), 1.0, atol=1e-5):
            raise ValueError("Support-detector reference vectors are not normalized.")

        extractor = _build_extractor(model)
        if int(extractor.output_shape[0][-1]) != support_vectors.shape[1]:
            raise ValueError("Support detector feature dimensions do not match.")

        self.metadata = metadata
        self.class_names = list(class_names)
        self.image_size = image_size
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.intercept = intercept
        self.gamma = gamma
        self.threshold = threshold
        self.extractor = extractor

    def check(self, image_path, prediction, image_size=(224, 224)):
        if self.extractor is None:
            raise RuntimeError("Supported-plant detector has not been loaded.")
        if tuple(image_size) != self.image_size:
            raise ValueError("Runtime image size differs from detector training.")
        confidence = float(prediction.get("confidence", 0.0))
        if not np.isfinite(confidence):
            raise ValueError("Prediction confidence is not finite.")
        if confidence < self.min_confidence:
            return {"is_known": False, "reason": "low_confidence"}
        if prediction.get("class_name") not in self.class_names:
            return {"is_known": False, "reason": "unsupported_class"}

        features, _ = self.extractor(
            preprocess_image(image_path, self.image_size), training=False
        )
        score = float(kernel_scores(
            np.asarray(features).reshape(1, -1), self.support_vectors,
            self.dual_coef, self.intercept, self.gamma,
        )[0])
        if not np.isfinite(score):
            raise ValueError("Support-detector score is not finite.")
        accepted = score >= self.threshold
        return {
            "is_known": bool(accepted),
            "reason": "supported" if accepted else "unsupported_or_uncertain",
            "support_score": score,
            "threshold": self.threshold,
        }

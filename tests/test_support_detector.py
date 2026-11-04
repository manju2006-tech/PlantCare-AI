import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from utils.support_detector import (
    FORMAT_VERSION, PREPROCESSING, SupportedPlantDetector, file_sha256,
    kernel_scores, normalize_features,
)


class FakeExtractor:
    output_shape = [(None, 3), (None, 1)]

    def __call__(self, image, training=False):
        return image[:, 0, 0, :] + 1.0, np.ones((1, 1))


class SupportDetectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model_path = self.folder / "disease.keras"
        self.model_path.write_bytes(b"fixed disease model")
        self.artifact = self.folder / "support.npz"
        self.classes = ["Rice___Blast"]
        self.metadata = {
            "format_version": FORMAT_VERSION, "preprocessing": PREPROCESSING,
            "image_size": [224, 224], "class_names": self.classes,
            "disease_model_sha256": file_sha256(self.model_path),
        }
        self.write_artifact()
        self.detector = SupportedPlantDetector(str(self.artifact))
        self.extractor_patch = patch(
            "utils.unknown_detector._build_extractor", return_value=FakeExtractor()
        )
        self.extractor_patch.start()
        self.addCleanup(self.extractor_patch.stop)

    def write_artifact(self, **overrides):
        values = dict(
            metadata=np.asarray(json.dumps(self.metadata)),
            support_vectors=np.asarray([[1., 0., 0.], [0., 1., 0.]]),
            dual_coef=np.asarray([2., -2.]), intercept=np.asarray(0.),
            gamma=np.asarray(2.), threshold=np.asarray(0.),
        )
        values.update(overrides)
        np.savez_compressed(self.artifact, **values)

    def load(self):
        self.detector.load(object(), self.classes, str(self.model_path))

    def test_unknown_is_rejected_even_with_100_percent_disease_confidence(self):
        self.load()
        image = self.folder / "unknown.png"
        Image.new("RGB", (10, 10), (0, 255, 0)).save(image)
        result = self.detector.check(
            str(image), {"class_name": self.classes[0], "confidence": 100.0}
        )
        self.assertFalse(result["is_known"])
        self.assertEqual(result["reason"], "unsupported_or_uncertain")

    def test_supported_features_are_accepted(self):
        self.load()
        image = self.folder / "known.png"
        Image.new("RGB", (10, 10), (255, 0, 0)).save(image)
        self.assertTrue(self.detector.check(
            str(image), {"class_name": self.classes[0], "confidence": 99.0}
        )["is_known"])

    def test_model_replacement_invalidates_detector(self):
        self.model_path.write_bytes(b"new disease model")
        with self.assertRaisesRegex(ValueError, "different model"):
            self.load()

    def test_label_reordering_invalidates_detector(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            self.detector.load(object(), ["Rice___Tungro"], str(self.model_path))

    def test_missing_and_invalid_artifacts_fail_closed(self):
        detector = SupportedPlantDetector(str(self.folder / "missing.npz"))
        with self.assertRaises(FileNotFoundError):
            detector.load(object(), self.classes, str(self.model_path))
        self.write_artifact(dual_coef=np.asarray([np.nan, 0.]))
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.load()

    def test_bad_features_and_predictions_are_rejected(self):
        for features in ([[0., 0.]], [[np.nan, 1.]], [[np.inf, 1.]]):
            with self.subTest(features=features), self.assertRaises(ValueError):
                normalize_features(features)
        self.load()
        with self.assertRaisesRegex(ValueError, "not finite"):
            self.detector.check("unused", {"confidence": float("nan")})
        with self.assertRaisesRegex(ValueError, "image size"):
            self.detector.check("unused", {}, (128, 128))
        self.assertFalse(self.detector.check("unused", {
            "class_name": self.classes[0], "confidence": 49.99
        })["is_known"])
        self.assertFalse(self.detector.check("unused", {
            "class_name": "unrecognized", "confidence": 100.
        })["is_known"])

    def test_portable_scores_match_sklearn_for_unseen_queries(self):
        from sklearn.svm import SVC
        rng = np.random.default_rng(14)
        features = rng.normal(size=(40, 7))
        labels = (features[:, 0] > 0).astype(int)
        classifier = SVC(C=10., gamma=2.).fit(normalize_features(features), labels)
        queries = rng.normal(size=(12, 7))
        scores = kernel_scores(
            queries, classifier.support_vectors_, classifier.dual_coef_[0],
            classifier.intercept_[0], classifier._gamma,
        )
        np.testing.assert_allclose(
            scores, classifier.decision_function(normalize_features(queries)), atol=1e-7
        )


class PredictionRejectionTests(unittest.TestCase):
    """Exercise the actual Flask endpoint without opening the user's database."""

    def setUp(self):
        from flask import Flask
        from flask_login import LoginManager, UserMixin
        from routes.prediction_routes import prediction_bp
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        app = Flask(__name__)
        app.config.update(
            TESTING=True, SECRET_KEY="test", UPLOAD_FOLDER=str(self.folder),
            REPORT_FOLDER=str(self.folder), ALLOWED_EXTENSIONS={"jpg"},
            IMAGE_SIZE=(224, 224), UNKNOWN_PLANT_CHECK=True,
        )
        login = LoginManager(app)

        class User(UserMixin):
            id = 7

            def to_dict(self):
                return {"username": "test"}

        login.user_loader(lambda user_id: User())
        app.register_blueprint(prediction_bp)
        self.app = app
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session["_user_id"] = "7"
            session["_fresh"] = True
        self.detector = Mock()
        self.app.extensions["unknown_plant_detector"] = self.detector
        self.prediction = SimpleNamespace(
            id=21, report_filename=None, to_dict=lambda: {"id": 21}
        )
        patches = {
            "ready": patch("routes.prediction_routes.model_loader.is_model_ready", return_value=True),
            "plant": patch("routes.prediction_routes.model_loader.predict_plant", return_value={"is_plant": True}),
            "predict": patch("routes.prediction_routes.model_loader.predict_image", return_value={
                "class_name": "Rice___Blast", "confidence": 100.
            }),
            "save": patch("routes.prediction_routes.save_prediction", return_value=self.prediction),
            "report": patch("routes.prediction_routes.generate_prediction_report", return_value="test.pdf"),
            "db": patch("routes.prediction_routes.db"),
        }
        self.mocks = {}
        for name, patcher in patches.items():
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def upload(self):
        image = io.BytesIO()
        Image.new("RGB", (20, 20), "green").save(image, format="JPEG")
        image.seek(0)
        return self.client.post("/predict", data={"image": (image, "leaf.jpg")})

    def assert_no_diagnosis(self):
        self.mocks["save"].assert_not_called()
        self.mocks["report"].assert_not_called()
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_rejection_returns_422_and_never_saves_diagnosis_or_pdf(self):
        self.detector.check.return_value = {"is_known": False}
        response = self.upload()
        self.assertEqual(response.status_code, 422)
        self.assertTrue(response.json["unknown_plant"])
        self.assertFalse(response.json["success"])
        self.assert_no_diagnosis()

    def test_detector_unavailable_prevents_prediction_from_being_saved(self):
        self.app.extensions["unknown_plant_detector"] = None
        with self.assertLogs(self.app.logger, level="ERROR"):
            self.assertEqual(self.upload().status_code, 503)
        self.assert_no_diagnosis()

    def test_detector_error_prevents_prediction_from_being_saved(self):
        self.detector.check.side_effect = ValueError("bad detector data")
        with self.assertLogs(self.app.logger, level="ERROR"):
            self.assertEqual(self.upload().status_code, 503)
        self.assert_no_diagnosis()

    def test_supported_scan_preserves_successful_report_workflow(self):
        self.detector.check.return_value = {"is_known": True}
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json["success"])
        self.assertEqual(response.json["redirect_url"], "/result/21")
        self.mocks["save"].assert_called_once()
        self.mocks["report"].assert_called_once()
        self.assertEqual(self.prediction.report_filename, "test.pdf")


if __name__ == "__main__":
    unittest.main()

"""Real-model upload regression checks, using an isolated temporary database.

Run after training/integration: .venv/bin/python -B tests/smoke_support_detector.py
The real application database, uploads, and reports are not used.
"""
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from config import Config

    with tempfile.TemporaryDirectory(prefix="plantcare-support-smoke-") as temporary:
        folder = Path(temporary)
        Config.SQLALCHEMY_DATABASE_URI = f"sqlite:///{folder / 'test.db'}"
        Config.UPLOAD_FOLDER = str(folder / "uploads")
        Config.REPORT_FOLDER = str(folder / "reports")
        Config.WTF_CSRF_ENABLED = False
        Config.SECRET_KEY = "temporary-test-session"
        Config.DEFAULT_ADMIN_USERNAME = "temporary-admin"
        Config.DEFAULT_ADMIN_PASSWORD = "TemporaryTest123"
        os.environ["FLASK_ENV"] = "development"

        from app import app
        from utils.db_utils import Prediction
        from utils.support_detector import SupportedPlantDetector

        detector = app.extensions["unknown_plant_detector"]
        assert isinstance(detector, SupportedPlantDetector), "Support detector failed at startup."
        client = app.test_client()
        response = client.post("/register", data={
            "username": "temporary_user", "email": "temporary@example.test",
            "password": "TemporaryTest123", "confirm_password": "TemporaryTest123",
        })
        assert response.status_code == 302, response.status_code
        response = client.get("/upload")
        assert response.status_code == 200
        assert b"unsupportedPlantModal" in response.data

        def upload(path):
            with path.open("rb") as image:
                return client.post("/predict", data={"image": (io.BytesIO(image.read()), "leaf.jpg")})

        # These exact photos reproduced the previous wheat-as-rice failure.
        for name in (
            "wheat_healthy/loh(42).JPG",
            "wheat_damaged/septoria/los(18).JPG",
            "wheat_damaged/strip_rust/lolr(16).JPG",
        ):
            response = upload(ROOT / "dataset/unknown_test" / name)
            assert response.status_code == 422, (name, response.status_code, response.json)
            assert response.json.get("unknown_plant") is True
            print("Rejected former false diagnosis:", name, flush=True)
        with app.app_context():
            assert Prediction.query.count() == 0
        assert not list(Path(Config.UPLOAD_FOLDER).iterdir())
        assert not list(Path(Config.REPORT_FOLDER).iterdir())

        # A rice original held out from the support classifier's fitting set.
        with (ROOT / "trained_model_support/sample_manifest.csv").open(newline="") as source:
            rice = next(row for row in csv.DictReader(source)
                        if row["category"] == "Rice___Blast" and row["split"] == "test")
        response = upload(ROOT / rice["path"])
        assert response.status_code == 200, (response.status_code, response.json)
        assert response.json.get("success") is True
        result = client.get(response.json["redirect_url"])
        assert result.status_code == 200
        with app.app_context():
            assert Prediction.query.count() == 1
            prediction = Prediction.query.first()
            assert prediction.report_filename, "Accepted scan did not generate its report."
            report_url = f"/result/{prediction.id}/report"
        assert client.get(report_url).status_code == 200
        print("Accepted supported rice; result, history record and PDF verified.", flush=True)
        print("Real-model upload regression checks passed using temporary data.", flush=True)


if __name__ == "__main__":
    main()

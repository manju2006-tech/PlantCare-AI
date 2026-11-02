
from pathlib import Path
import sys
import csv
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask
from config import Config
from utils import model_loader
from utils.unknown_detector import UnknownPlantDetector

UNKNOWN_DIR = PROJECT_ROOT / "dataset" / "unknown_test"
PROFILE_PATH = PROJECT_ROOT / "trained_model_57class_run2" / "unknown_profile.npz"
OUTPUT_PATH = PROJECT_ROOT / "unknown_image_evaluation.csv"

CATEGORIES = [
    "wheat_healthy",
    "wheat_damaged/septoria",
    "wheat_damaged/strip_rust",
    "other_plants",
    "unsupported_plants",
    "non_plants",
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Use the project's configured model paths and image size.
    app = Flask(__name__)
    app.config.from_object(Config)

    with app.app_context():
        model_loader.load_model(app)

    if not model_loader.is_model_ready():
        raise RuntimeError(
            "Model failed to load: " + str(model_loader.get_model_load_error())
        )

    # Load the same trained model used by model_loader into the detector.
    # This does not train or modify the model.
    import tensorflow as tf

    model = tf.keras.models.load_model(app.config["MODEL_PATH"])

    detector = UnknownPlantDetector(
        str(PROFILE_PATH),
        min_confidence=50.0,
    )
    detector.load(model)

    images = []
    for category in CATEGORIES:
        folder = UNKNOWN_DIR / category
        if not folder.is_dir():
            print(f"WARNING: Missing folder: {folder}")
            continue

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        # Limit UCI images to 100 for this first comparison.
        if category == "other_plants":
            files = files[:100]

        images.extend((category, path) for path in files)

    if not images:
        print("No images found. Check dataset/unknown_test.")
        return

    counts = defaultdict(lambda: {"accepted": 0, "rejected": 0, "errors": 0})
    rows = []

    print(f"Evaluating {len(images)} images...")
    print("No training or live application files will be changed.")

    for i, (category, path) in enumerate(images, 1):
        try:
            prediction = model_loader.predict_image(str(path))
            result = detector.check(str(path), prediction)

            accepted = bool(result["is_known"])
            counts[category]["accepted" if accepted else "rejected"] += 1

            rows.append({
                "category": category,
                "image": str(path.relative_to(PROJECT_ROOT)),
                "predicted_class": prediction["class_name"],
                "confidence_percent": prediction["confidence"],
                "accepted_as_supported": accepted,
                "reason": result.get("reason", ""),
                "distance": result.get("distance", ""),
                "threshold": result.get("threshold", ""),
                "nearest_class": result.get("nearest_class", ""),
            })

        except Exception as exc:
            counts[category]["errors"] += 1
            rows.append({
                "category": category,
                "image": str(path.relative_to(PROJECT_ROOT)),
                "predicted_class": "ERROR",
                "confidence_percent": "",
                "accepted_as_supported": "",
                "reason": str(exc),
                "distance": "",
                "threshold": "",
                "nearest_class": "",
            })

        if i % 25 == 0 or i == len(images):
            print(f"Processed {i}/{len(images)}")

    fields = [
        "category", "image", "predicted_class", "confidence_percent",
        "accepted_as_supported", "reason", "distance", "threshold",
        "nearest_class",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== RESULTS =====")
    for category in CATEGORIES:
        c = counts[category]
        total = c["accepted"] + c["rejected"]
        print(
            f"{category}: accepted {c['accepted']}, "
            f"rejected {c['rejected']}, errors {c['errors']}, "
            f"successful {total}"
        )

    print(f"\nCSV saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

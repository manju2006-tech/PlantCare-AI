
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COCO_IMAGE_DIR = Path.home() / "Downloads/non_plant_dataset/val2017"
ANNOTATION_FILE = Path.home() / "Downloads/annotations/instances_val2017.json"

OUTPUT_DIR = PROJECT_ROOT / "dataset/plant_binary_source/non_plant"

# COCO categories that represent non-plant objects
NON_PLANT_CATEGORIES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}

# Categories that indicate visible plants.
PLANT_CATEGORIES = {"potted plant"}

MAX_IMAGES = 1500

def main():
    if not COCO_IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image folder not found: {COCO_IMAGE_DIR}")

    if not ANNOTATION_FILE.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ANNOTATION_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading COCO annotations...")

    with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    category_names = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    selected_ids = set()

    # Only select images with at least one non-plant object
    # and no annotated potted plant.
    for annotation in data["annotations"]:
        category_name = category_names.get(annotation["category_id"])

        if category_name in PLANT_CATEGORIES:
            continue

        if category_name in NON_PLANT_CATEGORIES:
            selected_ids.add(annotation["image_id"])

    images = [
        image for image in data["images"]
        if image["id"] in selected_ids
    ]

    print(f"Eligible images found: {len(images)}")

    # Avoid overwriting existing files.
    existing = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}

    copied = 0
    missing = 0

    for image in tqdm(images, desc="Copying non-plant images"):
        if copied >= MAX_IMAGES:
            break

        filename = image["file_name"]
        source = COCO_IMAGE_DIR / filename
        destination = OUTPUT_DIR / f"coco_{filename}"

        if destination.name in existing:
            continue

        if not source.is_file():
            missing += 1
            continue

        shutil.copy2(source, destination)
        existing.add(destination.name)
        copied += 1

    print("\nSelection completed!")
    print(f"New images copied: {copied}")
    print(f"Missing source images: {missing}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original COCO images and existing test images were not modified.")

if __name__ == "__main__":
    main()

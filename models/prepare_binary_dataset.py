
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "dataset" / "plant_binary_source"
OUTPUT = ROOT / "dataset" / "plant_binary_classifier"

SEED = 42
SPLITS = ("train", "validation", "test")
RATIOS = (0.70, 0.15, 0.15)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def allocate_groups(groups):
    """Split groups without separating related iNaturalist observations."""
    rng = random.Random(SEED)
    groups = list(groups)
    rng.shuffle(groups)

    total = sum(len(items) for items in groups)
    targets = [total * ratio for ratio in RATIOS]
    counts = [0, 0, 0]
    result = {name: [] for name in SPLITS}

    for group in groups:
        index = max(
            range(3),
            key=lambda i: targets[i] - counts[i]
        )
        result[SPLITS[index]].extend(group)
        counts[index] += len(group)

    return result


def copy_images(images, label, prefix):
    """Copy images into the independent train/validation/test folders."""
    groups = defaultdict(list)

    for path, group_id in images:
        groups[group_id].append(path)

    split_groups = allocate_groups(groups.values())
    copied = {}

    for split, paths in split_groups.items():
        destination = OUTPUT / split / label
        destination.mkdir(parents=True, exist_ok=True)

        for path in paths:
            target = destination / f"{prefix}{path.name}"
            if target.exists():
                raise FileExistsError(target)

            shutil.copy2(path, target)

        copied[split] = len(paths)

    return copied


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(
            f"{OUTPUT} already contains files. "
            "Do not overwrite an existing dataset."
        )

    plant_folder = SOURCE / "plant"
    inat_folder = SOURCE / "inaturalist_download" / "images"
    nonplant_folder = SOURCE / "non_plant"

    # Existing 57-class plant images.
    existing_plant = [
        (p, p.name)
        for p in plant_folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # iNaturalist: group images by observation ID.
    metadata_path = SOURCE / "inaturalist_download" / "metadata.csv"
    inat_groups = defaultdict(list)

    with open(metadata_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filename = row["filename"]
            observation_id = row["observation_id"]
            path = inat_folder / filename

            if path.is_file():
                inat_groups[observation_id].append(path)

    inat_images = [
        (path, observation_id)
        for observation_id, paths in inat_groups.items()
        for path in paths
    ]

    # Read COCO annotations and exclude every image containing
    # an annotated potted plant.
    annotation_path = (
        Path.home() / "Downloads" / "annotations"
        / "instances_val2017.json"
    )

    with open(annotation_path, encoding="utf-8") as f:
        coco = json.load(f)

    category_names = {
        item["id"]: item["name"]
        for item in coco["categories"]
    }

    potted_plant_ids = {
        image_id
        for ann in coco["annotations"]
        if category_names.get(ann["category_id"]) == "potted plant"
        for image_id in [ann["image_id"]]
    }

    nonplant_images = []

    for path in nonplant_folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            image_id = int(path.stem.replace("coco_", ""))
        except ValueError:
            continue

        if image_id not in potted_plant_ids:
            nonplant_images.append((path, str(image_id)))

    print("Source counts:")
    print("Existing plant:", len(existing_plant))
    print("iNaturalist plant:", len(inat_images))
    print("COCO non-plant after exclusion:", len(nonplant_images))

    if len(existing_plant) != 1710:
        raise ValueError("Unexpected existing plant image count.")
    if len(inat_images) != 1000:
        raise ValueError("Unexpected iNaturalist image count.")
    if len(nonplant_images) != 1446:
        raise ValueError(
            "Expected 1446 COCO images after exclusion. "
            "Check the source folders and annotations."
        )

    print("\nCopying plant images...")
    pv_counts = copy_images(existing_plant, "plant", "pv__")
    inat_counts = copy_images(inat_images, "plant", "inat__")

    print("Copying non-plant images...")
    coco_counts = copy_images(nonplant_images, "non_plant", "coco__")

    print("\nFinal split counts:")
    for split in SPLITS:
        plants = pv_counts[split] + inat_counts[split]
        negatives = coco_counts[split]
        print(
            f"{split}: plant={plants}, "
            f"non_plant={negatives}, total={plants + negatives}"
        )

    print("\nDataset created at:", OUTPUT)
    print("Original source folders were not modified.")


if __name__ == "__main__":
    main()

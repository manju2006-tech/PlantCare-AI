
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_TRAIN = PROJECT_ROOT / "dataset/plant_dataset/PlantData/train"
SOURCE_VALID = PROJECT_ROOT / "dataset/plant_dataset/PlantData/valid"
OUTPUT = PROJECT_ROOT / "dataset/mobilenet_experiment"

SEED = 42
MANGO_TRAIN = 640
MANGO_VALID = 80
MANGO_TEST = 80

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_images(files, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for source in files:
        target = destination / source.name

        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Destination already exists: {target}"
            )

        target.symlink_to(source.resolve())


def main():
    if not SOURCE_TRAIN.is_dir() or not SOURCE_VALID.is_dir():
        raise FileNotFoundError(
            "Original train or validation directory is missing."
        )

    # Refuse to overwrite an existing experiment.
    for split in ("train", "validation", "test"):
        folder = OUTPUT / split
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing folder: {folder}. Create the three folders first."
            )
        if any(folder.iterdir()):
            raise RuntimeError(
                f"{folder} is not empty. Nothing was changed."
            )

    old_train_classes = {
        p.name for p in SOURCE_TRAIN.iterdir()
        if p.is_dir() and not p.name.startswith("Mango___")
    }
    old_valid_classes = {
        p.name for p in SOURCE_VALID.iterdir()
        if p.is_dir()
    }

    if len(old_train_classes) != 38 or len(old_valid_classes) != 38:
        raise RuntimeError(
            "Expected 38 original classes in both train and validation."
        )

    if old_train_classes != old_valid_classes:
        raise RuntimeError(
            "Original train and validation class names do not match."
        )

    mango_classes = sorted(
        p for p in SOURCE_TRAIN.glob("Mango___*") if p.is_dir()
    )
    if len(mango_classes) != 8:
        raise RuntimeError(
            f"Expected 8 Mango classes, found {len(mango_classes)}."
        )

    # Original 38 classes: train from train/, validate from valid/.
    for class_name in sorted(old_train_classes):
        train_files = get_images(SOURCE_TRAIN / class_name)
        valid_files = get_images(SOURCE_VALID / class_name)

        if not train_files or not valid_files:
            raise RuntimeError(
                f"Empty original class: {class_name}"
            )

        link_images(train_files, OUTPUT / "train" / class_name)
        link_images(valid_files, OUTPUT / "validation" / class_name)

        print(
            f"{class_name}: train={len(train_files)}, "
            f"validation={len(valid_files)}"
        )

    # Mango: independent, deterministic train/validation/test split.
    rng = random.Random(SEED)

    for class_dir in mango_classes:
        files = get_images(class_dir)

        if len(files) != 800:
            raise RuntimeError(
                f"{class_dir.name}: expected 800 images, found {len(files)}."
            )

        rng.shuffle(files)

        train_files = files[:MANGO_TRAIN]
        valid_files = files[MANGO_TRAIN:MANGO_TRAIN + MANGO_VALID]
        test_files = files[MANGO_TRAIN + MANGO_VALID:]

        link_images(train_files, OUTPUT / "train" / class_dir.name)
        link_images(valid_files, OUTPUT / "validation" / class_dir.name)
        link_images(test_files, OUTPUT / "test" / class_dir.name)

        print(
            f"{class_dir.name}: train={len(train_files)}, "
            f"validation={len(valid_files)}, test={len(test_files)}"
        )

    print("\nDataset preparation completed.")
    print("Expected classes: 46 (38 original + 8 Mango)")
    print("Mango totals: 5,120 train, 640 validation, 640 test")
    print(f"Dataset location: {OUTPUT}")
    print("Original images were not moved or modified.")
    print(
        "\nImportant: the Mango test set is held out from training. "
        "The original 38-class validation set is used for validation, "
        "not training."
    )


if __name__ == "__main__":
    main()

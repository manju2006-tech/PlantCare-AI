
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "dataset/plantcare_57class/train"
OUTPUT_DIR = PROJECT_ROOT / "dataset/plant_binary_source/plant"

TARGET_PER_CLASS = 30
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    total_copied = 0

    class_folders = sorted(
        folder for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    print(f"Plant classes found: {len(class_folders)}")

    for class_folder in class_folders:
        images = [
            file for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        selected = random.sample(
            images,
            min(TARGET_PER_CLASS, len(images))
        )

        copied = 0

        for image in selected:
            destination = OUTPUT_DIR / f"{class_folder.name}__{image.name}"

            if destination.exists():
                continue

            shutil.copy2(image, destination)
            copied += 1

        total_copied += copied

        print(f"{class_folder.name}: {copied} copied")

    print("\nPlant image collection completed!")
    print(f"Total new images copied: {total_copied}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original 57-class dataset was not modified.")

if __name__ == "__main__":
    main()

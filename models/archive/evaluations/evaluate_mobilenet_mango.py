
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "trained_model_mobilenet_experiment/best_model.keras"
CLASS_MAP_PATH = ROOT / "trained_model_mobilenet_experiment/class_names.json"
TEST_DIR = ROOT / "dataset/mobilenet_experiment/test"

BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

if not CLASS_MAP_PATH.is_file():
    raise FileNotFoundError(CLASS_MAP_PATH)

if not TEST_DIR.is_dir():
    raise FileNotFoundError(TEST_DIR)

with open(CLASS_MAP_PATH, "r") as f:
    class_names = json.load(f)

mango_classes = sorted(
    p.name for p in TEST_DIR.iterdir()
    if p.is_dir() and p.name.startswith("Mango___")
)

if len(mango_classes) != 8:
    raise RuntimeError(
        f"Expected 8 Mango test classes, found {len(mango_classes)}."
    )

# Load the best checkpoint, not the last-epoch model.
model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=mango_classes,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

def preprocess(images, labels):
    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(images, tf.float32)
    )
    return images, labels

test_ds = test_ds.map(preprocess)

all_true = []
all_pred = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    all_true.extend(labels.numpy().tolist())
    all_pred.extend(predictions.tolist())

all_true = np.asarray(all_true)
all_pred = np.asarray(all_pred)

mango_indices = [
    class_names.index(name) for name in mango_classes
]


# Convert test labels from local Mango indices (0–7)
# to their corresponding indices in the full 46-class model.
true_global = np.asarray([
    mango_indices[int(label)] for label in all_true
])

correct = int(np.sum(true_global == all_pred))
accuracy = correct / len(true_global) if len(true_global) else 0.0


print("\n========== MANGO TEST RESULTS ==========")
print(f"Test images: {len(all_true)}")
print(f"Correct predictions: {correct}")
print(f"Overall accuracy: {accuracy * 100:.2f}%")

print("\nPer-class accuracy:")
for i, class_name in enumerate(mango_classes):
    mask = all_true == i
    total = int(mask.sum())
    class_correct = int((all_pred[mask] == mango_indices[i]).sum())
    class_accuracy = (
        100 * class_correct / total if total else 0.0
    )
    print(
        f"{class_name}: {class_correct}/{total} "
        f"({class_accuracy:.2f}%)"
    )

print("\nMost common predicted classes for Mango test images:")
unique, counts = np.unique(all_pred, return_counts=True)
for idx in np.argsort(counts)[::-1][:10]:
    print(f"{class_names[int(unique[idx])]}: {int(counts[idx])}")

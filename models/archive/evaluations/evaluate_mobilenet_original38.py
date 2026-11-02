
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "trained_model_mobilenet_experiment/best_model.keras"
)
CLASS_MAP_PATH = (
    ROOT / "trained_model_mobilenet_experiment/class_names.json"
)
VALID_DIR = ROOT / "dataset/plant_dataset/PlantData/valid"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

with open(CLASS_MAP_PATH, "r") as f:
    class_names = json.load(f)

original_classes = sorted(
    p.name for p in VALID_DIR.iterdir() if p.is_dir()
)

if len(original_classes) != 38:
    raise RuntimeError(
        f"Expected 38 original classes, found {len(original_classes)}"
    )

if not all(name in class_names for name in original_classes):
    raise RuntimeError("Some validation classes are missing from the model.")

model = tf.keras.models.load_model(MODEL_PATH)

dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    class_names=original_classes,
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

dataset = dataset.map(preprocess)

true_global = []
pred_global = []

for images, labels in dataset:
    probabilities = model.predict(images, verbose=0)
    predicted_indices = np.argmax(probabilities, axis=1)

    true_global.extend([
        class_names.index(original_classes[int(i)])
        for i in labels.numpy()
    ])
    pred_global.extend(predicted_indices.tolist())

true_global = np.asarray(true_global)
pred_global = np.asarray(pred_global)

correct = int(np.sum(true_global == pred_global))
total = len(true_global)

print("\n===== ORIGINAL 38-CLASS EVALUATION =====")
print(f"Images evaluated: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Overall accuracy: {100 * correct / total:.2f}%")

print("\nPer-class accuracy:")
for name in original_classes:
    class_index = class_names.index(name)
    mask = true_global == class_index
    class_total = int(mask.sum())
    class_correct = int(
        np.sum(pred_global[mask] == class_index)
    )
    accuracy = (
        100 * class_correct / class_total
        if class_total else 0
    )
    print(f"{name}: {class_correct}/{class_total} ({accuracy:.2f}%)")

print("\nNote: predictions of Mango classes count as incorrect")
print("when the actual image belongs to an original class.")

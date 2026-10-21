
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "trained_model_mobilenet_experiment/best_model.keras"
CLASS_PATH = ROOT / "trained_model_mobilenet_experiment/class_names.json"
VALID_DIR = ROOT / "dataset/mobilenet_experiment/validation"

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    class_names=class_names,
    image_size=(224, 224),
    batch_size=16,
    shuffle=False,
    label_mode="int",
)

y_true, y_pred = [], []

for images, labels in ds:
    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(images, tf.float32)
    )
    probabilities = model.predict(images, verbose=0)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(np.argmax(probabilities, axis=1).tolist())

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_true, y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=3,
    zero_division=0,
))

cm = confusion_matrix(
    y_true, y_pred,
    labels=list(range(len(class_names))),
)

output = ROOT / "trained_model_mobilenet_experiment/confusion_matrix.csv"
np.savetxt(output, cm, delimiter=",", fmt="%d")

print(f"\nConfusion matrix saved to: {output}")

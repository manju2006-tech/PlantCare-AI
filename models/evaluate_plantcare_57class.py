import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "plantcare_57class"
MODEL_DIR = ROOT / "trained_model_57class_run2"
TEST_DIR = DATASET_DIR / "test"
RESULTS_DIR = MODEL_DIR / "evaluation"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

model_path = MODEL_DIR / "best_model.keras"
names_path = MODEL_DIR / "class_names.json"

if not model_path.is_file():
    raise FileNotFoundError(f"Model not found: {model_path}")

with names_path.open() as f:
    saved_classes = json.load(f)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

if test_ds.class_names != saved_classes:
    raise ValueError(
        "Test class order does not match class_names.json.\n"
        f"Dataset: {test_ds.class_names}\n"
        f"Saved: {saved_classes}"
    )

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
test_ds = test_ds.map(
    lambda images, labels: (preprocess(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

print("\nLoading best model...")
model = tf.keras.models.load_model(model_path)

print("\nEvaluating held-out test dataset...")
loss, model_accuracy = model.evaluate(test_ds, verbose=1)

true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)
probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4%}")
print(f"Model.evaluate accuracy: {model_accuracy:.4%}")

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(saved_classes))),
    target_names=saved_classes,
    output_dict=True,
    zero_division=0,
)
report_text = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(saved_classes))),
    target_names=saved_classes,
    zero_division=0,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(report).transpose().to_csv(
    RESULTS_DIR / "classification_report.csv"
)

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(saved_classes))),
)
pd.DataFrame(
    matrix, index=saved_classes, columns=saved_classes
).to_csv(RESULTS_DIR / "confusion_matrix.csv")

with (RESULTS_DIR / "summary.txt").open("w") as f:
    f.write(f"Test images: {len(true_labels)}\n")
    f.write(f"Test loss: {loss:.6f}\n")
    f.write(f"Test accuracy: {accuracy:.6%}\n\n")
    f.write(report_text)

print("\nPer-class results:")
print(report_text)
print(f"\nResults saved in: {RESULTS_DIR}")

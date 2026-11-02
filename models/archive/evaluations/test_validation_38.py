import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "trained_model_38_backup",
                          "plant_disease_model.h5")
CLASS_PATH = os.path.join(ROOT, "trained_model_38_backup",
                          "class_indices.json")
VALID_DIR = os.path.join(ROOT, "dataset", "plant_dataset",
                         "PlantData", "valid")

print("Loading model...", flush=True)
model = load_model(MODEL_PATH, compile=False)

with open(CLASS_PATH) as f:
    index_to_class = json.load(f)

gen = ImageDataGenerator().flow_from_directory(
    VALID_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="sparse",
    shuffle=False
)

correct = 0
total = 0
class_correct = {}
class_total = {}
batches = len(gen)

print(f"Testing {gen.samples} images in {batches} batches...", flush=True)

for step in range(batches):
    images, labels = next(gen)

    # Direct model call avoids repeated predict() overhead
    probs = model(images, training=False).numpy()
    predictions = np.argmax(probs, axis=1)

    for actual_idx, pred_idx in zip(labels.astype(int), predictions):
        actual = next(
            name for name, idx in gen.class_indices.items()
            if idx == actual_idx
        )
        predicted = index_to_class[str(int(pred_idx))]

        total += 1
        class_total[actual] = class_total.get(actual, 0) + 1

        if predicted == actual:
            correct += 1
            class_correct[actual] = class_correct.get(actual, 0) + 1

    if (step + 1) % 20 == 0 or step + 1 == batches:
        print(
            f"Progress: {total}/{gen.samples} images "
            f"({100 * total / gen.samples:.1f}%)",
            flush=True
        )

print("\nPer-class accuracy:")
for name in sorted(class_total):
    hits = class_correct.get(name, 0)
    count = class_total[name]
    print(f"{name}: {hits}/{count} ({100 * hits / count:.1f}%)")

print(f"\nOverall accuracy: {100 * correct / total:.2f}%")
print(f"Correct: {correct}/{total}")
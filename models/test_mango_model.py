import os
import json
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model_incremental",
    "plant_disease_model_46_finetuned.h5"
)

CLASS_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model_incremental",
    "class_indices_46.json"
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "plant_dataset",
    "PlantData",
    "train"
)

IMAGE_SIZE = (224, 224)

MANGO_CLASSES = [
    "Mango___Anthracnose",
    "Mango___Bacterial_Canker",
    "Mango___Cutting_Weevil",
    "Mango___Die_Back",
    "Mango___Gall_Midge",
    "Mango___healthy",
    "Mango___Powdery_Mildew",
    "Mango___Sooty_Mould",
]


def predict_image(model, image_path):
    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(
        arr,
        verbose=0
    )[0]

    index = int(np.argmax(predictions))
    confidence = float(predictions[index]) * 100

    return index, confidence


print("=" * 70)
print("PLANTCARE AI - MANGO MODEL TEST")
print("=" * 70)

print("\nLoading model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_PATH, "r") as f:
    index_to_class = json.load(f)

print("Model loaded successfully.")
print(f"Model output classes: {len(index_to_class)}")

print("\n" + "=" * 70)
print("TESTING MANGO CLASSES")
print("=" * 70)

total = 0
correct = 0

random.seed(42)

for class_name in MANGO_CLASSES:

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    random.shuffle(files)

    # Test 10 images from each Mango class
    files = files[:10]

    class_correct = 0

    for filename in files:

        image_path = os.path.join(
            class_dir,
            filename
        )

        predicted_index, confidence = predict_image(
            model,
            image_path
        )

        predicted_class = index_to_class[
            str(predicted_index)
        ]

        print(
            f"Actual: {class_name} | "
            f"Predicted: {predicted_class} | "
            f"Confidence: {confidence:.2f}%"
        )

        total += 1

        if predicted_class == class_name:
            correct += 1
            class_correct += 1

    accuracy = (
        class_correct / len(files)
    ) * 100

    print(
        f"{class_name:<35} "
        f"{class_correct}/{len(files)} "
        f"({accuracy:.1f}%)"
    )

print("\n" + "=" * 70)

overall_accuracy = (
    correct / total
) * 100

print(
    f"Mango test accuracy: "
    f"{overall_accuracy:.2f}%"
)

print(
    f"Correct: {correct}/{total}"
)

print("=" * 70)
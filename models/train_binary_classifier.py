
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "plant_binary_classifier"
OUTPUT_DIR = ROOT / "trained_model_binary"

MODEL_PATH = OUTPUT_DIR / "plant_binary.keras"
METADATA_PATH = OUTPUT_DIR / "binary_metadata.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_dataset(split, shuffle):
    folder = DATASET_DIR / split

    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        class_names=["non_plant", "plant"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# MODEL CREATION
# --------------------------------------------------

def build_model():
    inputs = layers.Input(shape=(224, 224, 3))

    # Convert raw pixel values from 0-255 to -1 to 1.
    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
    )(inputs)

    # Augmentation is active only during training.
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)

    # Separate ImageNet-pretrained MobileNetV2 backbone.
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    # Output is the probability that the image is a plant.
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="plant_probability",
    )(x)

    model = Model(inputs, outputs, name="plant_binary_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


# --------------------------------------------------
# THRESHOLD SELECTION
# --------------------------------------------------

def find_best_threshold(labels, probabilities):
    best_threshold = 0.50
    best_score = -1.0

    for threshold in np.arange(0.20, 0.81, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        # Positive class = plant (1)
        true_plant = labels == 1
        true_nonplant = labels == 0

        plant_recall = (
            np.sum((predictions == 1) & true_plant)
            / max(np.sum(true_plant), 1)
        )

        nonplant_recall = (
            np.sum((predictions == 0) & true_nonplant)
            / max(np.sum(true_nonplant), 1)
        )

        balanced_accuracy = (
            plant_recall + nonplant_recall
        ) / 2

        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = float(threshold)

    return best_threshold, best_score


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_threshold(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    plant_recall = tp / max(tp + fn, 1)
    nonplant_recall = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(labels), 1)

    print(f"Accuracy:          {accuracy * 100:.2f}%")
    print(f"Plant recall:      {plant_recall * 100:.2f}%")
    print(f"Non-plant recall:  {nonplant_recall * 100:.2f}%")
    print(f"False plant accepts (FP): {fp}")
    print(f"False plant rejects (FN): {fn}")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        "accuracy": round(float(accuracy), 4),
        "plant_recall": round(float(plant_recall), 4),
        "nonplant_recall": round(float(nonplant_recall), 4),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------

def main():
    print("=" * 60)
    print("PLANTCARE AI - BINARY CLASSIFIER TRAINING")
    print("=" * 60)

    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    if MODEL_PATH.exists():
        raise FileExistsError(
            f"Model already exists: {MODEL_PATH}. "
            "Move or rename it before starting a new training run."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset("train", shuffle=True)
    val_ds = load_dataset("validation", shuffle=False)
    test_ds = load_dataset("test", shuffle=False)

    print("\nClass mapping:")
    print("0 = non_plant")
    print("1 = plant")

    # Calculate balanced class weights.
    train_labels = np.concatenate(
        [labels.numpy().reshape(-1) for _, labels in train_ds]
    ).astype(int)

    counts = np.bincount(train_labels, minlength=2)
    total = len(train_labels)

    class_weights = {
        0: total / (2 * max(counts[0], 1)),
        1: total / (2 * max(counts[1], 1)),
    }

    print("\nTraining class counts:", counts.tolist())
    print("Class weights:", class_weights)

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting training...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint rather than relying on the final epoch.
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nSelecting threshold using validation data...")

    val_labels = np.concatenate(
        [labels.numpy().reshape(-1) for _, labels in val_ds]
    ).astype(int)

    val_probabilities = model.predict(val_ds, verbose=0).reshape(-1)

    threshold, validation_score = find_best_threshold(
        val_labels,
        val_probabilities,
    )

    print(f"Selected threshold: {threshold:.2f}")
    print(
        f"Validation balanced accuracy: "
        f"{validation_score * 100:.2f}%"
    )

    print("\nFinal evaluation on untouched test data:")

    test_labels = np.concatenate(
        [labels.numpy().reshape(-1) for _, labels in test_ds]
    ).astype(int)

    test_probabilities = model.predict(test_ds, verbose=0).reshape(-1)

    test_metrics = evaluate_threshold(
        test_labels,
        test_probabilities,
        threshold,
    )

    metadata = {
        "model_name": "PlantCare AI Binary Classifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "image_size": list(IMAGE_SIZE),
        "class_mapping": {
            "0": "non_plant",
            "1": "plant",
        },
        "plant_threshold": round(threshold, 2),
        "validation_balanced_accuracy": round(
            validation_score, 4
        ),
        "test_metrics": test_metrics,
        "backbone": "MobileNetV2 ImageNet pretrained",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nTraining completed.")
    print("Binary model:", MODEL_PATH)
    print("Metadata:", METADATA_PATH)
    print("Existing 57-class model was not modified.")


if __name__ == "__main__":
    main()

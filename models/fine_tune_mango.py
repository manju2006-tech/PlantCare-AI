
import os
import sys
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.data_preprocessing import (
    create_data_generators,
    compute_generator_class_weights,
)

# Use the newly trained 46-class model as the starting point.
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model_incremental",
    "plant_disease_model_46.h5",
)

CLASS_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model_incremental",
    "class_indices_46.json",
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "incremental_46",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_model_incremental",
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "plant_disease_model_46_finetuned.h5",
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-5
SEED = 42


def main():
    print("=" * 65)
    print("PLANTCARE AI - STAGE 2 FINE-TUNING")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Starting model not found: {MODEL_PATH}"
        )

    if not os.path.exists(CLASS_PATH):
        raise FileNotFoundError(
            f"Class mapping not found: {CLASS_PATH}"
        )

    with open(CLASS_PATH, "r") as f:
        index_to_class = json.load(f)

    if len(index_to_class) != 46:
        raise RuntimeError(
            f"Expected 46 classes, found {len(index_to_class)}"
        )

    train_gen, val_gen = create_data_generators(
        dataset_dir=DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        seed=SEED,
    )

    if len(train_gen.class_indices) != 46:
        raise RuntimeError(
            "Expected 46 classes in the fine-tuning dataset."
        )

    # Ensure class order matches the saved model mapping.
    saved_mapping = {
        int(index): name
        for index, name in index_to_class.items()
    }

    for name, index in train_gen.class_indices.items():
        if saved_mapping.get(index) != name:
            raise RuntimeError(
                f"Class mapping mismatch at index {index}: "
                f"{name} != {saved_mapping.get(index)}"
            )

    print("\nLoading 46-class model...")
    model = load_model(MODEL_PATH, compile=False)

    # Freeze the whole feature extractor first.
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze the final convolutional block and its
    # associated BatchNormalization layers.
    unfreeze_names = {
        "conv2d_6",
        "batch_normalization_6",
        "conv2d_7",
        "batch_normalization_7",
    }

    found = set()

    for layer in model.layers:
        if layer.name in unfreeze_names:
            layer.trainable = True
            found.add(layer.name)

    # Train the 46-class output layer too.
    model.layers[-1].trainable = True

    if found != unfreeze_names:
        raise RuntimeError(
            "Could not identify all intended final convolutional "
            f"layers. Found: {sorted(found)}. "
            "No training was started."
        )

    print("\nTrainable layers:")
    for layer in model.layers:
        if layer.trainable:
            print(" -", layer.name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            OUTPUT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    class_weights = compute_generator_class_weights(train_gen)

    print("\nClass weights:")

    index_to_class = {
        index: name
        for name, index in train_gen.class_indices.items()
    }

    for class_index, weight in sorted(class_weights.items()):
        class_name = index_to_class.get(class_index, str(class_index))
        print(f"{class_name}: {weight:.4f}")


    print("\nStarting fine-tuning...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("The original 38-class backup will not be changed.")

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the best restored weights to the separate output file.
    model.save(OUTPUT_PATH)

    print("\n" + "=" * 65)
    print("FINE-TUNING COMPLETE")
    print(f"Fine-tuned model: {OUTPUT_PATH}")
    print("Original model and class mapping remain untouched.")
    print("=" * 65)


if __name__ == "__main__":
    main()

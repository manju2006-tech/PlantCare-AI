"""
Incrementally expand the existing 38-class PlantCare AI model
to 46 classes by adding the 8 Mango classes.

IMPORTANT:
- Does NOT modify train_model.py
- Uses the existing trained 38-class model as the starting point
- Creates a temporary balanced dataset using symbolic links
- Keeps the original 38-class model safe
"""

import os
import sys
import json
import random
import shutil

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from models.cnn_model import build_cnn_model, compile_model
from models.data_preprocessing import (
    create_data_generators,
    compute_generator_class_weights,
    save_class_indices,
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

OLD_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model_38_backup",
    "plant_disease_model.h5"
)

OLD_CLASS_INDICES_PATH = os.path.join(
    PROJECT_ROOT,
    "trained_model_38_backup",
    "class_indices.json"
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "plant_dataset",
    "PlantData",
    "train"
)

TEMP_DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "incremental_46"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "trained_model_incremental"
)

NEW_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "plant_disease_model_46.h5"
)

NEW_CLASS_INDICES_PATH = os.path.join(
    OUTPUT_DIR,
    "class_indices_46.json"
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

# We don't need hundreds/thousands of old images per class
# because the existing model already learned those classes.
OLD_IMAGES_PER_CLASS = 100

# Use all 800 images from each Mango class.
MANGO_IMAGES_PER_CLASS = 800

EPOCHS = 2

RANDOM_SEED = 42


# ---------------------------------------------------------
# CREATE TEMPORARY BALANCED DATASET
# ---------------------------------------------------------

def create_balanced_dataset():

    print("\n" + "=" * 70)
    print("CREATING INCREMENTAL DATASET")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    if os.path.exists(TEMP_DATASET_DIR):
        print("Removing old temporary incremental dataset...")
        shutil.rmtree(TEMP_DATASET_DIR)

    os.makedirs(TEMP_DATASET_DIR, exist_ok=True)

    class_dirs = sorted(
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    )

    print(f"Total classes found: {len(class_dirs)}")

    if len(class_dirs) != 46:
        raise RuntimeError(
            f"Expected 46 classes, but found {len(class_dirs)}."
        )

    for class_name in class_dirs:

        source_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        target_dir = os.path.join(
            TEMP_DATASET_DIR,
            class_name
        )

        os.makedirs(target_dir, exist_ok=True)

        images = []

        for filename in os.listdir(source_dir):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                images.append(filename)

        random.shuffle(images)

        if class_name.startswith("Mango___"):
            limit = MANGO_IMAGES_PER_CLASS
        else:
            limit = min(
                OLD_IMAGES_PER_CLASS,
                len(images)
            )

        selected_images = images[:limit]

        for filename in selected_images:

            source_file = os.path.join(
                source_dir,
                filename
            )

            target_file = os.path.join(
                target_dir,
                filename
            )

            os.symlink(
                source_file,
                target_file
            )

        print(
            f"{class_name}: "
            f"{len(selected_images)} images"
        )

    print("\nTemporary dataset created successfully.")
    print(f"Location: {TEMP_DATASET_DIR}")


# ---------------------------------------------------------
# LOAD OLD MODEL
# ---------------------------------------------------------

def load_old_model():

    print("\n" + "=" * 70)
    print("LOADING EXISTING 38-CLASS MODEL")
    print("=" * 70)

    if not os.path.exists(OLD_MODEL_PATH):
        raise FileNotFoundError(
            f"Old model not found:\n{OLD_MODEL_PATH}"
        )

    if not os.path.exists(OLD_CLASS_INDICES_PATH):
        raise FileNotFoundError(
            f"Old class mapping not found:\n"
            f"{OLD_CLASS_INDICES_PATH}"
        )

    old_model = load_model(
        OLD_MODEL_PATH,
        compile=False
    )

    with open(OLD_CLASS_INDICES_PATH, "r") as f:
        old_index_to_class = json.load(f)

    old_class_to_index = {
        class_name: int(index)
        for index, class_name in old_index_to_class.items()
    }

    print(
        f"Old model output classes: "
        f"{len(old_class_to_index)}"
    )

    return old_model, old_class_to_index


# ---------------------------------------------------------
# CREATE 46 CLASS MODEL AND TRANSFER WEIGHTS
# ---------------------------------------------------------

def create_incremental_model(
    old_model,
    old_class_to_index,
    new_class_to_index
):

    print("\n" + "=" * 70)
    print("CREATING 46-CLASS MODEL")
    print("=" * 70)

    new_model = build_cnn_model(
        input_shape=(224, 224, 3),
        num_classes=46
    )

    # -----------------------------------------------------
    # Copy all compatible layers except final classifier
    # -----------------------------------------------------

    old_layers = old_model.layers
    new_layers = new_model.layers

    copied_layers = 0

    for old_layer, new_layer in zip(
        old_layers[:-1],
        new_layers[:-1]
    ):

        old_weights = old_layer.get_weights()
        new_weights = new_layer.get_weights()

        if not old_weights:
            continue

        compatible = (
            len(old_weights) == len(new_weights)
            and all(
                old_w.shape == new_w.shape
                for old_w, new_w in zip(
                    old_weights,
                    new_weights
                )
            )
        )

        if compatible:
            new_layer.set_weights(old_weights)
            copied_layers += 1

    print(
        f"Copied weights from {copied_layers} "
        f"existing layers."
    )

    # -----------------------------------------------------
    # Transfer old 38-class classifier weights
    # -----------------------------------------------------

    old_output_layer = old_model.layers[-1]
    new_output_layer = new_model.layers[-1]

    old_kernel, old_bias = old_output_layer.get_weights()

    new_kernel, new_bias = new_output_layer.get_weights()

    # Copy old class weights into matching new class positions.
    for class_name, old_index in old_class_to_index.items():

        new_index = new_class_to_index[class_name]

        new_kernel[:, new_index] = old_kernel[:, old_index]
        new_bias[new_index] = old_bias[old_index]

    new_output_layer.set_weights(
        [new_kernel, new_bias]
    )

    print(
        "Transferred existing 38-class "
        "classifier weights."
    )

    # -----------------------------------------------------
    # Freeze feature extractor
    # -----------------------------------------------------

    for layer in new_model.layers[:-1]:
        layer.trainable = False

    new_model.layers[-1].trainable = True

    print(
        "Feature extractor: FROZEN"
    )

    print(
        "46-class output layer: TRAINABLE"
    )

    return new_model


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("PLANTCARE AI - INCREMENTAL 38 → 46 TRAINING")
    print("=" * 70)

    # 1. Create small balanced dataset
    create_balanced_dataset()

    # 2. Load old model
    old_model, old_class_to_index = load_old_model()

    # 3. Create data generators
    train_gen, val_gen = create_data_generators(
        dataset_dir=TEMP_DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        seed=RANDOM_SEED,
    )

    print(
        f"\nDetected classes: "
        f"{len(train_gen.class_indices)}"
    )

    if len(train_gen.class_indices) != 46:
        raise RuntimeError(
            "The incremental dataset does not contain 46 classes."
        )

    new_class_to_index = train_gen.class_indices

    # 4. Load/create incremental model
    model = create_incremental_model(
        old_model,
        old_class_to_index,
        new_class_to_index
    )

    # 5. Compile with small learning rate
    model = compile_model(
        model,
        learning_rate=1e-4
    )

    model.summary()

    # 6. Class weights
    class_weights = compute_generator_class_weights(
        train_gen
    )

    # 7. Output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # 8. Callbacks
    callbacks = [

        ModelCheckpoint(
            NEW_MODEL_PATH,
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
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 9. Train
    print("\n" + "=" * 70)
    print("STARTING INCREMENTAL TRAINING")
    print("=" * 70)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # 10. Save final model
    # 10. Keep the best checkpoint saved by ModelCheckpoint.
    # Do not overwrite it with the final epoch's model.
    print(f"Best checkpoint retained at: {NEW_MODEL_PATH}")

    # 11. Save class mapping
    save_class_indices(
        train_gen.class_indices,
        NEW_CLASS_INDICES_PATH
    )

    # 12. Evaluate
    val_loss, val_accuracy = model.evaluate(
        val_gen,
        verbose=0
    )

    print("\n" + "=" * 70)
    print("INCREMENTAL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Validation accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Model saved to:\n"
        f"{NEW_MODEL_PATH}"
    )

    print(
        f"Class mapping saved to:\n"
        f"{NEW_CLASS_INDICES_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
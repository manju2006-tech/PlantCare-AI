"""
PlantVillage dataset loading, augmentation, and class-index bookkeeping.

Expected dataset layout (standard PlantVillage / Kaggle "color" layout) -
one folder per class, containing that class's leaf images:

    dataset/PlantVillage/
        Apple___Apple_scab/
            image_0001.jpg
            image_0002.jpg
            ...
        Apple___Black_rot/
            ...
        Tomato___healthy/
            ...

Download options:
  - Kaggle: "PlantVillage Dataset" (emmarex/plantdisease)
  - Kaggle: "New Plant Diseases Dataset" (vipoooool/new-plant-diseases-dataset)

Extract either dataset so that each disease class is its own subfolder
directly inside `dataset_dir`, then point train_model.py at that directory.
"""
import os
import json

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(dataset_dir, image_size=(224, 224), batch_size=32,
                            validation_split=0.2, seed=42):
    """
    Returns (train_generator, val_generator) with an 80/20 train/validation
    split by default.

    IMPORTANT: no rescale=1./255 is applied here - the CNN itself contains a
    Rescaling(1/255) layer (see models/cnn_model.py), so raw 0-255 pixel
    values are intentionally passed straight through. This guarantees
    training-time and inference-time preprocessing can never drift apart.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Download the PlantVillage dataset from Kaggle "
            "(e.g. 'emmarex/plantdisease' or "
            "'vipoooool/new-plant-diseases-dataset') and extract it so each "
            "disease class is a subfolder directly inside this directory."
        )

    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode='nearest',
        validation_split=validation_split,
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=seed,
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=seed,
        shuffle=False,
    )

    return train_generator, val_generator


def compute_generator_class_weights(train_generator):
    """
    Computes balanced class weights from the training generator's label
    distribution, so classes with fewer images don't get under-learned.
    Safe to pass straight into model.fit(class_weight=...).
    """
    classes = train_generator.classes  # array of int labels, one per sample
    unique_classes = np.unique(classes)
    weights = compute_class_weight(
        class_weight='balanced', classes=unique_classes, y=classes
    )
    return {int(cls): float(w) for cls, w in zip(unique_classes, weights)}


def save_class_indices(class_indices, output_path):
    """
    Keras generators expose class_indices as {class_name: index}. We invert
    it to {index: class_name} (as strings, for valid JSON keys) because
    that's the direction needed at inference time: predicted index ->
    disease class name.
    """
    index_to_class = {str(v): k for k, v in class_indices.items()}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(index_to_class, f, indent=2)
    return index_to_class


def load_class_indices(input_path):
    with open(input_path, 'r') as f:
        return json.load(f)
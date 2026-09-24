"""
Train the plant disease CNN on the PlantVillage dataset and save all
artifacts the Flask app needs at inference time:

    trained_model/plant_disease_model.h5   - trained Keras model
    trained_model/class_indices.json       - {index: class_name} mapping
    trained_model/model_metadata.json      - training run metadata
    trained_model/training_history.png     - accuracy/loss curves

Usage:
    python models/train_model.py --dataset_dir dataset/PlantVillage --epochs 25

Run this from the project root (or any directory - paths are resolved
relative to this script's location, not the current working directory).
"""
import os
import sys
import json
import argparse
from datetime import datetime

# Make the project root importable when this script is run directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')  # no display needed - just save PNGs
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

from models.cnn_model import build_cnn_model, compile_model
from models.data_preprocessing import (
    create_data_generators, compute_generator_class_weights, save_class_indices
)

TRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_model')
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, 'plant_disease_model.h5')
CLASS_INDICES_PATH = os.path.join(TRAINED_MODEL_DIR, 'class_indices.json')
METADATA_PATH = os.path.join(TRAINED_MODEL_DIR, 'model_metadata.json')
HISTORY_PLOT_PATH = os.path.join(TRAINED_MODEL_DIR, 'training_history.png')
TRAINING_LOG_PATH = os.path.join(TRAINED_MODEL_DIR, 'training_log.csv')


def parse_args():
    parser = argparse.ArgumentParser(description='Train the plant disease CNN.')
    parser.add_argument(
        '--dataset_dir', type=str,
        default=os.path.join(PROJECT_ROOT, 'dataset', 'PlantVillage'),
        help='Path to the PlantVillage dataset (one subfolder per class).'
    )
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224, help='Square image side length.')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--validation_split', type=float, default=0.2)
    return parser.parse_args()


def plot_training_history(history, output_path):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

    image_size = (args.image_size, args.image_size)

    print(f"\n{'='*70}")
    print("PLANT DISEASE CNN - TRAINING")
    print(f"{'='*70}")
    print(f"Dataset directory : {args.dataset_dir}")
    print(f"Image size        : {image_size}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Epochs            : {args.epochs}")
    print(f"Learning rate     : {args.learning_rate}")
    print(f"{'='*70}\n")

    # --- Data ---
    train_gen, val_gen = create_data_generators(
        dataset_dir=args.dataset_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )
    num_classes = len(train_gen.class_indices)
    print(f"Detected {num_classes} classes: {list(train_gen.class_indices.keys())}\n")

    class_weights = compute_generator_class_weights(train_gen)

    # --- Model ---
    model = build_cnn_model(input_shape=(image_size[0], image_size[1], 3), num_classes=num_classes)
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', patience=8,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
        CSVLogger(TRAINING_LOG_PATH, append=False),
    ]

    # --- Train ---
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint already saved the best model to MODEL_PATH, but save
    # again explicitly in case training finished without ever beating the
    # initial checkpoint (e.g. very short debug runs).
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # --- Save class indices (index -> class name) ---
    index_to_class = save_class_indices(train_gen.class_indices, CLASS_INDICES_PATH)
    print(f"Class indices saved to: {CLASS_INDICES_PATH}")

    # --- Final evaluation on the validation set ---
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    print(f"\nFinal validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Final validation loss    : {val_loss:.4f}")

    # --- Save metadata ---
    metadata = {
        'trained_at': datetime.utcnow().isoformat() + 'Z',
        'num_classes': num_classes,
        'classes': index_to_class,
        'image_size': list(image_size),
        'epochs_run': len(history.history.get('accuracy', [])),
        'final_val_accuracy': round(float(val_accuracy), 4),
        'final_val_loss': round(float(val_loss), 4),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {METADATA_PATH}")

    # --- Save training curves ---
    plot_training_history(history, HISTORY_PLOT_PATH)
    print(f"Training curves saved to: {HISTORY_PLOT_PATH}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - the Flask app will auto-load this model on next start.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
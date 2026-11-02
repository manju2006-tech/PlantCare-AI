from pathlib import Path
import json
import math
import tensorflow as tf

# --------------------------------------------------
# PlantCare AI: separate 57-class training experiment
# Does NOT modify the existing live model.
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "plantcare_57class"
OUTPUT_DIR = ROOT / "trained_model_57class_run2"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Safety checks: all splits must exist and be non-empty.
for split in ("train", "validation", "test"):
    path = DATASET_DIR / split
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {path}")

train_dir = DATASET_DIR / "train"
validation_dir = DATASET_DIR / "validation"
test_dir = DATASET_DIR / "test"

class_names = sorted(
    p.name for p in train_dir.iterdir() if p.is_dir()
)

if len(class_names) != 57:
    raise RuntimeError(
        f"Expected 57 classes, found {len(class_names)}."
    )

# Ensure identical class folders in every split.
for split_dir in (validation_dir, test_dir):
    names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if names != class_names:
        raise RuntimeError(
            f"Class-folder mismatch in {split_dir}."
        )

# Refuse to overwrite a previous experiment.
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise RuntimeError(
        f"{OUTPUT_DIR} already contains files. "
        "Nothing was overwritten. Choose a new output folder "
        "or inspect the existing experiment first."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets using one consistent class order.
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

# Calculate balanced class weights from training counts.
counts = []
for name in class_names:
    folder = train_dir / name
    count = sum(1 for p in folder.iterdir() if p.is_file())
    if count == 0:
        raise RuntimeError(f"No training images found for {name}.")
    counts.append(count)

total_images = sum(counts)
num_classes = len(class_names)
class_weight = {
    i: total_images / (num_classes * counts[i])
    for i in range(num_classes)
}

# Save the exact class order and training counts for reproducibility.
(OUTPUT_DIR / "class_names.json").write_text(
    json.dumps(class_names, indent=2)
)
(OUTPUT_DIR / "class_counts.json").write_text(
    json.dumps(
        {name: counts[i] for i, name in enumerate(class_names)},
        indent=2,
    )
)
(OUTPUT_DIR / "class_weights.json").write_text(
    json.dumps(
        {class_names[i]: class_weight[i] for i in range(num_classes)},
        indent=2,
    )
)

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

def prepare(images, labels):
    images = preprocess(tf.cast(images, tf.float32))
    return images, labels

train_ds = train_ds.map(
    prepare, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

validation_ds = validation_ds.map(
    prepare, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

# Fresh MobileNetV2 initialized with ImageNet weights.
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(
    num_classes, activation="softmax", name="predictions"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(OUTPUT_DIR / "training_log.csv")
    ),
]

print("\n===== PlantCare AI 57-class training =====")
print("Dataset:", DATASET_DIR)
print("Output:", OUTPUT_DIR)
print("Classes:", num_classes)
print("Training images:", total_images)
print("Validation images:", sum(
    1 for p in validation_dir.rglob("*") if p.is_file()
))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Backbone: MobileNetV2 (ImageNet, frozen)")
print("Class weighting: enabled")
print("Existing live model: untouched")

model.summary()

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
)

# Save the final epoch separately from the best checkpoint.
model.save(OUTPUT_DIR / "final_model.keras")

print("\nTraining finished.")
print("Best checkpoint:", OUTPUT_DIR / "best_model.keras")
print("Final model:", OUTPUT_DIR / "final_model.keras")
print("Class names:", OUTPUT_DIR / "class_names.json")
print("Review training_log.csv before evaluating the test set.")


from pathlib import Path
import json
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset/mobilenet_experiment"
OUTPUT_DIR = PROJECT_ROOT / "trained_model_mobilenet_experiment"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 5
SEED = 42

# Use CPU-friendly settings and reproducible data ordering.
tf.keras.utils.set_random_seed(SEED)

for split in ("train", "validation"):
    if not (DATASET_DIR / split).is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MobileNetV2 expects preprocessing to scale pixel values to [-1, 1].
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

class_names = train_ds.class_names

validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "validation",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

if len(class_names) != 46:
    raise RuntimeError(
        f"Expected 46 classes, found {len(class_names)}."
    )

# Save the class order used by this model.
with open(OUTPUT_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

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

# Load pretrained feature extractor and freeze it initially.
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(
    len(class_names), activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(OUTPUT_DIR / "training_log.csv")
    ),
]

print("\nStarting MobileNetV2 experiment...")
print("Training images:", len(train_ds) * BATCH_SIZE)
print("Validation images:", len(validation_ds) * BATCH_SIZE)
print("Classes:", len(class_names))
print("Output folder:", OUTPUT_DIR)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(OUTPUT_DIR / "final_model.keras")

print("\nTraining finished.")
print("Best checkpoint:", OUTPUT_DIR / "best_model.keras")
print("Final model:", OUTPUT_DIR / "final_model.keras")
print("Class names:", OUTPUT_DIR / "class_names.json")

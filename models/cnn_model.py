"""
CNN architecture for plant disease image classification.

A custom convolutional neural network (not a pretrained backbone) sized to
train comfortably on the PlantVillage dataset on a single GPU or even CPU
for a demo run. Uses an internal Rescaling layer so raw 0-255 pixel images
can be fed in directly - no separate normalization step is required at
inference time.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape=(224, 224, 3), num_classes=23):
    """
    Builds a 4-block CNN:
      [Conv -> BatchNorm -> Conv -> BatchNorm -> MaxPool -> Dropout] x 4
      -> GlobalAveragePooling -> Dense -> Dropout -> Softmax output

    Each block doubles the filter count (32 -> 64 -> 128 -> 256) so the
    network learns increasingly abstract leaf-texture/lesion features while
    spatial resolution shrinks. GlobalAveragePooling (instead of Flatten)
    keeps the parameter count manageable and reduces overfitting risk on a
    moderately sized dataset.
    """
    model = models.Sequential(name='plant_disease_cnn')

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255.0))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.20))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.35))

    # Classification head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', name='predictions'))

    return model


def compile_model(model, learning_rate=1e-3):
    """Compile with Adam optimizer, categorical crossentropy, and accuracy metric."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


if __name__ == '__main__':
    # Quick sanity check: build and print the model summary.
    m = build_cnn_model(input_shape=(224, 224, 3), num_classes=23)
    m = compile_model(m)
    m.summary()
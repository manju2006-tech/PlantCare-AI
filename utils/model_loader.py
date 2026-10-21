"""
Singleton loader for the trained plant disease CNN, used by the Flask app
for real-time inference. The model is loaded exactly once at application
startup (see app.py) and reused across every request - not reloaded from
disk on every prediction.
"""
import os

import json
import numpy as np
from PIL import Image

_model = None
_index_to_class = None
_image_size = (224, 224)
_model_load_error = None


class ModelNotLoadedError(Exception):
    """Raised when a prediction is requested but no trained model is available."""
    pass


def load_model(app):
    """
    Called once at Flask app startup. Loads the .h5 model and
    class_indices.json from paths defined in config.py.

    If the model hasn't been trained yet, this logs a clear warning
    instead of crashing the whole application - the /predict route will
    return a friendly, actionable error until a model is trained and the
    app is restarted.
    """
    global _model, _index_to_class, _image_size, _model_load_error

    _image_size = app.config['IMAGE_SIZE']
    model_path = app.config['MODEL_PATH']
    class_indices_path = app.config['CLASS_INDICES_PATH']

    if not os.path.exists(model_path) or not os.path.exists(class_indices_path):
        _model_load_error = (
            "No trained model found. Run "
            "'python models/train_model.py --dataset_dir dataset/PlantVillage' "
            "to train and save a model, then restart the app."
        )
        app.logger.warning(_model_load_error)
        return

    try:
        import tensorflow as tf
        from models.data_preprocessing import load_class_indices

        _model = tf.keras.models.load_model(model_path)
        with open(class_indices_path, "r") as f:
            class_map = json.load(f)

        if isinstance(class_map, list):
            _index_to_class = {
                str(i): name for i, name in enumerate(class_map)
            }
        else:
            _index_to_class = class_map
        _model_load_error = None
        app.logger.info(
            f"Plant disease model loaded successfully from {model_path} "
            f"({len(_index_to_class)} classes)."
        )
    except Exception as exc:  # noqa: BLE001 - want to catch and log any load failure
        _model_load_error = f"Failed to load model: {exc}"
        app.logger.error(_model_load_error)
        _model = None


def is_model_ready() -> bool:
    return _model is not None and _index_to_class is not None


def get_model_load_error():
    return _model_load_error



def _preprocess_image(image_path):
    """
    Prepares an image for MobileNetV2 inference.
    Pixel values are scaled to [-1, 1].
    """
    import tensorflow as tf

    img = Image.open(image_path).convert('RGB')
    img = img.resize((_image_size[1], _image_size[0]))

    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    return arr



def predict_image(image_path, top_k=3):
    """
    Runs inference on a single image file.

    Returns:
        {
            'class_name': str,        # e.g. "Tomato___Late_blight"
            'confidence': float,      # 0-100
            'top_predictions': [ {'class_name': str, 'confidence': float}, ... ]
        }

    Raises:
        ModelNotLoadedError if no trained model is currently loaded.
    """
    if not is_model_ready():
        raise ModelNotLoadedError(_model_load_error or "Model is not loaded.")

    arr = _preprocess_image(image_path)
    predictions = _model.predict(arr, verbose=0)[0]  # shape: (num_classes,)

    top_indices = predictions.argsort()[::-1][:top_k]
    top_predictions = [
        {
            'class_name': _index_to_class[str(int(idx))],
            'confidence': round(float(predictions[idx]) * 100, 2),
        }
        for idx in top_indices
    ]

    best_idx = int(top_indices[0])
    return {
        'class_name': _index_to_class[str(best_idx)],
        'confidence': round(float(predictions[best_idx]) * 100, 2),
        'top_predictions': top_predictions,
    }
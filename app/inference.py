import tensorflow as tf
import numpy as np
from .model_loader import ensure_model_dir

MODEL_DIR = ensure_model_dir()
model = tf.keras.models.load_model(MODEL_DIR)

def predict_image(batch: np.ndarray) -> dict:
    p_real = float(model(batch, training=False).numpy()[0][0])
    p_fake = 1.0 - p_real

    if p_fake >= p_real:
        return {"label": "fake", "confidence": p_fake, "scores": {"fake": p_fake, "real": p_real}}
    return {"label": "real", "confidence": p_real, "scores": {"fake": p_fake, "real": p_real}}
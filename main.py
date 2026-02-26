import os
from io import BytesIO
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from keras.layers import TFSMLayer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

IMG_SIZE = (299, 299)


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probability_real: float


def _parse_cors_allow_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _resolve_model_path() -> tuple[Path, str]:
    model_dir = os.getenv("FFD_MODEL_DIR")
    if model_dir:
        model_path = Path(model_dir).expanduser().resolve()
        loader = "keras" if model_path.is_file() and model_path.suffix == ".keras" else "savedmodel"
        return model_path, loader

    hf_model_repo = os.getenv("HF_MODEL_REPO")
    if hf_model_repo:
        hf_token = os.getenv("HF_TOKEN")
        snapshot_path = snapshot_download(
            repo_id=hf_model_repo,
            token=hf_token,
            local_dir="/tmp/fake_face_model",
            local_dir_use_symlinks=False,
        )
        snapshot_root = Path(snapshot_path).resolve()
        keras_file = snapshot_root / "xception_face_detector.keras"
        if keras_file.exists():
            return keras_file, "keras"
        return snapshot_root, "savedmodel"

    # Default: use the trained model inside fake-face-detection project.
    return (
        (Path(__file__).resolve().parent.parent / "fake-face-detection" / "fake_face_model").resolve(),
        "savedmodel",
    )


MODEL_PATH, MODEL_LOADER = _resolve_model_path()

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model directory not found at '{MODEL_PATH}'. Set FFD_MODEL_DIR or HF_MODEL_REPO."
    )

if MODEL_LOADER == "keras":
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
else:
    model = TFSMLayer(str(MODEL_PATH), call_endpoint="serving_default")

app = FastAPI(title="Fake Face Detection API", version="1.0.0")

cors_allow_origins = _parse_cors_allow_origins()
cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() in {
    "1",
    "true",
    "yes",
}

if "*" in cors_allow_origins and cors_allow_credentials:
    raise RuntimeError("CORS_ALLOW_CREDENTIALS=true requires explicit CORS_ALLOW_ORIGINS values.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_path": str(MODEL_PATH)}


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    return np.expand_dims(img_array, axis=0)


@tf.function
def call_model(input_tensor: tf.Tensor):
    if MODEL_LOADER == "keras":
        return model(input_tensor, training=False)
    return model(input_tensor)


def predict_fake_face(image: Image.Image) -> PredictionResponse:
    img_array = preprocess_image(image)
    output = call_model(img_array)

    if isinstance(output, dict):
        prob_real = float(list(output.values())[0][0][0].numpy())
    else:
        prob_real = float(output[0][0].numpy())

    label = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = prob_real if prob_real >= 0.5 else (1 - prob_real)
    confidence_percent = confidence * 100
    probability_real_percent = prob_real * 100

    return PredictionResponse(
        label=label,
        confidence=round(confidence_percent, 2),
        probability_real=round(probability_real_percent, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pil_img = Image.open(BytesIO(payload))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    return predict_fake_face(pil_img)

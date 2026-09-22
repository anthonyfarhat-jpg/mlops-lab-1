from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from food11.data import CATEGORY_NAMES

MODEL_URI = "models:/food11@champion"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
IMAGE_SIZE = (128, 128)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(contents: bytes) -> np.ndarray:
    """Convert an uploaded image to the normalized NCHW tensor used in training."""
    with Image.open(io.BytesIO(contents)) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    normalized = (pixels - MEAN) / STD
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


def probabilities_from_prediction(prediction: Any) -> np.ndarray:
    """Turn the PyTorch model's logits into a one-dimensional probability vector."""
    logits = np.asarray(prediction, dtype=np.float32).squeeze()
    if logits.ndim != 1 or logits.size != len(CATEGORY_NAMES):
        raise ValueError(f"Expected {len(CATEGORY_NAMES)} logits, received shape {logits.shape}")

    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    yield


app = FastAPI(title="Food-11 Classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str | float]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="The uploaded file must be an image")

    try:
        model_input = preprocess_image(await file.read())
        probabilities = probabilities_from_prediction(app.state.model.predict(model_input))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    predicted_index = int(np.argmax(probabilities))
    return {
        "category": CATEGORY_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
    }

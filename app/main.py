from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.inference import get_transform, load_model, predict

MODEL_PATH = Path("artifacts/best_model.pt")

model: torch.nn.Module | None = None
class_names: list[str] = []
device: torch.device | None = None
transform: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, device, transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(MODEL_PATH, device)
    class_names = cast(list[str], checkpoint["class_names"])
    transform = get_transform(checkpoint)

    yield # everything above gets executed before the app starts, everything below gets executed after the app stops


app = FastAPI(
    title="CIFAR-10 Image Classifier API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "device": str(device),
    }


@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    top_k: int = 3,
) -> dict[str, Any]:
    if model is None or device is None or transform is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    if top_k < 1 or top_k > len(class_names):
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between 1 and {len(class_names)}.",
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload a valid image file.",
        )

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be decoded as an image.",
        ) from exc

    predictions, inference_time_ms = predict(
        model=model,
        image=image,
        transform=transform,
        class_names=class_names,
        device=device,
        top_k=top_k,
    )

    return {
        "filename": file.filename,
        "device": str(device),
        "inference_time_ms": inference_time_ms,
        "predictions": predictions,
    }
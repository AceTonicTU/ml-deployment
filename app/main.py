from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

app = FastAPI(
    title="Vision Inference Service",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": False,
    }


@app.post("/inspect-image")
async def inspect_image(file: UploadFile = File(...)) -> dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image uploads are supported.",
        )

    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    return {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }
# ML Vision Deployment

A FastAPI service that serves a fine-tuned ResNet-18 CIFAR-10 classifier.
The service accepts an uploaded image and returns top-k predicted classes,
confidence scores, and inference latency.

## Stack

- Python
- PyTorch 2.14.0 with CUDA 13.2
- Torchvision
- FastAPI and Uvicorn
- Docker Desktop with WSL2 GPU passthrough
- NVIDIA GPU inference

## Local API

```powershell
conda activate ml-deployment
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## GPU Docker

### Build

```powershell
docker build -t ml-vision-api:gpu-0.1.0 .
```

### Run

```powershell
docker run --rm --name ml-vision-api `
  --gpus all `
  -p 8000:8000 `
  ml-vision-api:gpu-0.1.0
```

Open:

```text
http://127.0.0.1:8000/docs
```

Expected health response:

```json
{
  "status": "ok",
  "device": "cuda",
  "model_loaded": "true"
}
```

## API

### `GET /health`

Returns whether the service is running, whether a model is loaded, and the selected inference device.

### `POST /predict`

Upload an image as `multipart/form-data`.

Query parameter:

- `top_k`: Number of predictions to return, from 1 to 10.

Example response:

```json
{
  "filename": "example.jpg",
  "device": "cuda",
  "inference_time_ms": 4.2,
  "predictions": [
    {
      "label": "truck",
      "confidence": 0.59
    }
  ]
}
```

## Important limitation

The model was trained on CIFAR-10 and can only output its ten predefined
classes. Predictions on real-world images are deployment tests and should not
be interpreted as reliable semantic classification.
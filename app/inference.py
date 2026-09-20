from pathlib import Path

import torch

import time
from PIL import Image

import torchvision
from torchvision.transforms import v2

def build_model(num_classes: int) -> torch.nn.Module:
    """Builds a ResNet18 model with a custom output layer for the specified number of classes."""
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(checkpoint_path: Path, device: torch.device,) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(num_classes=checkpoint["num_classes"])

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint

def predict(model: torch.nn.Module, image: Image.Image, transform: v2.Compose, class_names: list[str], device: torch.device, top_k: int) -> tuple[list[dict], float]:
    """Predicts the class of the given image using the provided model and transformation."""
    image_tensor = transform(image).unsqueeze(0).to(device) # unsqueeze because model expects a batch dimension

    start_time = time.perf_counter()

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidences, indices = probabilities.topk(top_k, dim=1)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    results = [
        {
            "label": class_names[int(idx.item())],
            "confidence": float(conf.item()),
        }
        for conf, idx in zip(confidences[0], indices[0])
    ]

    return results, inference_time_ms

def get_transform(checkpoint: dict) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(checkpoint["input_size"][1:]),  # Resize to (H, W)
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=checkpoint["normalization_mean"],
            std=checkpoint["normalization_std"],
        ),
    ])


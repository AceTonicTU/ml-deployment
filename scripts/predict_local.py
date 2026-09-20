from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from app.inference import get_transform, load_model, predict

DEFAULT_CHECKPOINT = Path("models/cifar10_resnet18_v0.1.0.pth")

def load_image(image_path: Path) -> Image.Image:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"File is not a supported image: {image_path}") from e

def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CIFAR-10 ResNet18 model inference on a single image.")
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to return.",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top_k must be atleast 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model(args.checkpoint_path, device)
    transform = get_transform(checkpoint)
    image = load_image(args.image_path)

    results, inference_time_ms = predict(
        model=model,
        image=image,
        transform=transform,
        class_names=checkpoint["class_names"],
        device=device,
        top_k=min(args.top_k, checkpoint["num_classes"]),
    )

    print(f"Model: {checkpoint['model_name']} v{checkpoint['model_version']}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Image: {args.image_path}")
    print(f"Original Image Size: {image.size}")
    print(f"Device: {device}")
    print(f"Inference Time in milliseconds (ms): {inference_time_ms:.2f}")
    print(f"Top predictions:")
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['label']} (confidence: {result['confidence']:.4f})")

if __name__ == "__main__":
    main()
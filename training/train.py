# %%
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.models import ResNet18_Weights

import numpy as np
import matplotlib.pyplot as plt

# %%
import random
import numpy as np
import torch

# Hyperparameters
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
NUM_WORKERS = 2

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# %%
transform_cpu = v2.Compose([
    v2.ToImage(),
])

transform_gpu_train = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(),
    v2.RandomCrop(32, padding=4),
    v2.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

transform_gpu_eval = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

# %%
# Training loop
def train_model(model, trainloader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            inputs = transform_gpu_train(inputs) # GPU Batch Train-Augmentation
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

# %%
# Evaluation loop
def evaluate_model(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.inference_mode():
        for inputs, labels in testloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            inputs = transform_gpu_eval(inputs) # GPU Batch Eval-Augmentation
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(testloader)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# %%
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(SEED)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # load dataset
    trainset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform_cpu
    )

    testset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform_cpu
    )

    # DataLoaders
    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0
    )

    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    # load ResNet18 model
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)

    # only train classifier? (False) is much worse here
    for parameter in model.parameters():
        parameter.requires_grad = True

    # modify last layer for CIFAR-10 (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)

    model = model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    train_model(
        model=model,
        trainloader=trainloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device
    )

    test_loss, test_accuracy = evaluate_model(
        model=model,
        testloader=testloader,
        criterion=criterion,
        device=device
    )

    checkpoint = {
        "model_name": "resnet18",
        "model_version": "0.1.0",
        "num_classes": 10,
        "class_names": trainset.classes,
        "normalization_mean": (0.5, 0.5, 0.5),
        "normalization_std": (0.5, 0.5, 0.5),
        "epoch": EPOCHS,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "model_state_dict": model.state_dict(),
    }

    checkpoint_path = MODELS_DIR / "cifar10_resnet18_v0.1.0.pth"
    torch.save(checkpoint, checkpoint_path)

    print(f"Saved checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    main()
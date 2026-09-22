from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Food-11 classifier with MLflow tracking")
    parser.add_argument("--dataset", choices=["mini", "processed"], default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def get_dataset_root(dataset_name: str) -> Path:
    dataset_name = "food11_processed_mini" if dataset_name == "mini" else "food11_processed"
    root = Path("data") / dataset_name
    if not root.exists():
        raise FileNotFoundError(f"Dataset does not exist: {root}")
    return root


def build_dataloaders(dataset_root: Path, batch_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = datasets.ImageFolder(root=str(dataset_root / "training"), transform=transform)
    val_set = datasets.ImageFolder(root=str(dataset_root / "validation"), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_set.class_to_idx


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    total_items = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_items += labels.size(0)

    return total_loss / max(total_items, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return loss, accuracy


def main() -> None:
    args = parse_args()
    dataset_root = get_dataset_root(args.dataset)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, class_to_idx = build_dataloaders(dataset_root, args.batch_size)
    model = build_model(len(class_to_idx))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_classes": len(class_to_idx),
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")

        final_val_loss, final_val_accuracy = evaluate(model, val_loader, criterion, device)
        mlflow.log_metric("final_val_accuracy", final_val_accuracy, step=args.epochs)
        mlflow.log_metric("final_val_loss", final_val_loss, step=args.epochs)

        example_input = torch.randn(1, 3, 128, 128, device=device)
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example_input,
            serialization_format="pickle",
        )

        print(f"Final validation accuracy: {final_val_accuracy:.4f}")


if __name__ == "__main__":
    main()

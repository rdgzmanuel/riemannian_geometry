# src/hdm05_grassmann/training/train_baseline.py

from __future__ import annotations

import argparse

import torch
from src.data.datasets import HDM05WindowsDataset
from src.data.data_loader import get_dataloaders
from src.models.baselines import MLPBaseline
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose
from .eval import evaluate_epoch
from .losses import get_classification_loss
from .utils import get_device, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento baseline MLP en HDM05")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument(
        "--checkpoint", type=str, default="experiments/checkpoints/baseline/mlp.pt"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset: usamos un único dataset y lo partimos en train/val
    # para que compartan label2idx.
    # ------------------------------------------------------------------
    ds = HDM05WindowsDataset()  # usa rutas por defecto

    seed = 42
    batch_size = 32

    train_loader, val_loader, test_loader = get_dataloaders(
        ds,
        batch_size=batch_size,
        seed=seed
    )

    # Para averiguar T y d, cogemos una muestra
    x0, _ = next(iter(train_loader)) # x0: (T, d)
    _, T, d = x0.shape
    num_classes = len(ds.label2idx)

    print(f"MLPBaseline:{T, d}T, d) input_dim={T * d}, num_classes={num_classes}")

    model = MLPBaseline(input_dim=T*d, num_classes=num_classes).to(device)
    criterion = get_classification_loss("ce")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        # for x, y in train_loader:
        #     x = x.to(device)  # (B, T, d)
        #     y = y.to(device)

        for x, y in train_loader:
            # X_list = [x.to(device) for x in X_list]
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # x: (B, T, d)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)


        train_loss = total_loss / max(total_samples, 1)

        val_loss, val_acc = evaluate_epoch(model, val_loader, device, criterion)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc * 100:.2f}%"
        )

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.checkpoint, model, optimizer, epoch, best_val_acc)

    print(f"Entrenamiento finalizado. Mejor val_acc={best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

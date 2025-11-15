# src/hdm05_grassmann/training/train_baseline.py

from __future__ import annotations

import argparse

import torch
from src.data.datasets import HDM05WindowsDataset
from src.models.baselines import MLPBaseline
from torch.utils.data import DataLoader, random_split

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
    print(f"🔧 Device: {device}")

    # ------------------------------------------------------------------
    # Dataset: usamos un único dataset y lo partimos en train/val
    # para que compartan label2idx.
    # ------------------------------------------------------------------
    full_ds = HDM05WindowsDataset()  # usa rutas por defecto

    n_total = len(full_ds)
    n_val = int(args.val_split * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Para averiguar T y d, cogemos una muestra
    x0, y0 = full_ds[0]  # x0: (T, d)
    T, d = x0.shape
    num_classes = len(full_ds.label2idx)

    print(f"MLPBaseline: input_dim={T * d}, num_classes={num_classes}")

    model = MLPBaseline(input_dim=T * d, num_classes=num_classes).to(device)
    criterion = get_classification_loss("ce")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)  # (B, T, d)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)  # (B, C)
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

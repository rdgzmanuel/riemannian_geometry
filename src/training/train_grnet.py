# src/hdm05_grassmann/training/train_grnet.py

from __future__ import annotations

import argparse

import torch
from src.data.datasets import HDM05GrassmannDataset
from src.data.data_loader import get_dataloaders
from src.models.grnet import GrassmannNetGeomstats
from torch.utils.data import DataLoader, random_split

from .eval import evaluate_epoch
from .losses import get_classification_loss
from .utils import get_device, save_checkpoint, set_seed

from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento GrNet (Geomstats) en HDM05"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/grnet/grnet_geomstats.pt",
    )
    parser.add_argument("--fr_layers", type=str, default="16,8")
    parser.add_argument("--hidden_dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.5)
    return parser.parse_args()


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Usando dispositivo: {device}")
    fr_layers = parse_int_list(args.fr_layers)
    hidden_dims = parse_int_list(args.hidden_dims)

    ds = HDM05GrassmannDataset()
    seed = args.seed
    batch_size = args.batch_size

    writer = SummaryWriter(log_dir="runs/grnet_geomstats")

    train_loader, val_loader, test_loader = get_dataloaders(
        ds,
        batch_size=batch_size,
        seed=seed
    )

    U0, y0 = ds[0]  # U0: (d, p)
    d, p_in = U0.shape
    num_classes = len(ds.label2idx)

    print(
        f"GrassmannNetGeomstats: d={d}, p_in={p_in}, fr_layers={fr_layers}, num_classes={num_classes}"
    )

    model = GrassmannNetGeomstats(
        d=d,
        p_in=p_in,
        num_classes=num_classes,
        fr_layers=fr_layers,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    criterion = get_classification_loss("ce")
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,   # regularización L2 suave
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for U, y in train_loader:
            U = U.to(device)  # (B, d, p)
            y = y.to(device)
            
            optimizer.zero_grad()
            logits = model(U)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)

        train_loss = total_loss / max(total_samples, 1)
        val_loss, val_acc = evaluate_epoch(model, val_loader, device, criterion)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.checkpoint, model, optimizer, epoch, best_val_acc)

        scheduler.step()

    writer.close()

    print(f"Finalizado. Mejor val_acc={best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.datasets import HDM05SPDDataset
from src.data.data_loader import get_dataloaders, geom_collate
from src.models.spdnet import SPDNet, StiefelSGD
from src.training.utils import get_device, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SPDNetGeomstats con HDM05-SPD"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proj_dim", type=int, default=[70, 50, 30])
    parser.add_argument("--metric", type=str, default="log-euclidean",
                        choices=["log-euclidean", "affine"])
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/checkpoints/spd/spdnet_geom.pt")
    return parser.parse_args()


def train_step(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)      # (B, d, d)
        y = y.to(device)      # (B,)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def val_step(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def test_step(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.eval()
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_acc = correct / max(total, 1)
    return avg_acc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device = {device}")

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    ds = HDM05SPDDataset()

    seed = args.seed
    batch_size = args.batch_size

    train_loader, val_loader, test_loader = get_dataloaders(
        ds,
        batch_size=batch_size,
        seed=seed,
    )

    # Para averiguar T y d, cogemos una muestra
    x0, _ = next(iter(train_loader))  # x0: (T, d)
    _, d_in, _ = x0.shape
    num_classes = len(ds.label2idx)

    print(
        f"SPDNetGeomstats: d_in={d_in}) input_dim=({d_in}×{d_in}), num_classes={num_classes}"
    )

    # ----------------------------------------------------------
    # Modelo + Optimizer
    # ----------------------------------------------------------
    model = SPDNet(
        input_dim=d_in,
        hidden_dims=args.proj_dim,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = StiefelSGD(model.parameters(), lr=args.lr)

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    # best_val_acc = 0.0
    # best_state = None

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        train_loss, train_acc = train_step(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = val_step(model, val_loader, criterion, device)

        # print(
        #     f"[Epoch {epoch:03d}] "
        #     f"train: loss={train_loss:.4f} acc={train_acc*100:5.2f}% | "
        #     f"val: loss={val_loss:.4f} acc={val_acc*100:5.2f}%"
        # )

        tqdm.write(
            f"train: loss={train_loss:.4f} acc={train_acc*100:5.2f}% | "
            f"val: loss={val_loss:.4f} acc={val_acc*100:5.2f}%"
        )

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        #     save_checkpoint(args.checkpoint, model, optimizer, epoch, best_val_acc)

    # Save model
    save_checkpoint(args.checkpoint, model, optimizer, epoch, val_acc)
    # ----------------------------------------------------------
    # Evaluación final (test)
    # ----------------------------------------------------------
    test_acc = test_step(model, test_loader, device)
    print(f"[TEST] acc={test_acc*100:.2f}%")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import json
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from src.data.datasets import HDM05SPDDataset
from src.data.data_loader import get_dataloaders
from src.models.spdnet import BiMapLayer, SPDNet, StiefelSGD
from src.training.utils import (
    get_device,
    save_checkpoint,
    set_seed,
    load_resume_checkpoint,
    load_metrics_json,
    save_metrics_json
)


from src.manifolds.spd_ops import compute_P_matrix, retraction_stiefel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SPDNetGeomstats con HDM05-SPD"
    )
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proj_dim", nargs="+", type=int,
                        default=[70, 50, 30])
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/checkpoints/spd/spdnet_geom.pt")
    parser.add_argument("--json_metrics", type=str,
                        default="experiments/checkpoints/spd/spdnet_metrics.json")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda el entrenamiento desde un checkpoint si existe",
    )
    return parser.parse_args()


def train_step(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lr: float,
):
    """
    Train one training epoch for the model
    Args:
    - model (nn.Module): model
    - dataloader (Dataloader): train dataloader
    - optimizer (torch.optim)
    - criterion (torch.nn.CrossEntropyLoss)
    - device (torch.device)
    Returns:
    - (float): average loss of the epoch, average accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)      # (B, d, d)
        y = y.to(device)      # (B,)

        optimizer.zero_grad()
        logits, _ = model(X)
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
    """
    Perform one validation epoch
    Args:
    - model (nn.Module): model
    - dataloader (Dataloader): validation dataloader
    - criterion (torch.nn.CrossEntropyLoss)
    - device (torch.device)
    Returns:
    - tuple with average_loss, accuracy

    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits, _ = model(X)
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
    """
    Perform one evaluation epoch of the best model
    Args:
    - model (nn.Module): model
    - loader (Dataloader): test dataloader
    - device (torch.device)
    Returns:
    - Test accuracy (float)

    """
    model.eval()
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits, _ = model(X)
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
        f"SPDNetGeomstats: d_in={d_in} input_dim=({d_in}×{d_in}), num_classes={num_classes}"
    )

    # ----------------------------------------------------------
    # Modelo + Optimizer
    # ----------------------------------------------------------
    model = SPDNet(
        d_in=d_in,
        proj_dim=args.proj_dim,
        num_classes=num_classes,
    ).to(device)

    # Mark BiMap weights as being on Stiefel manifold
    for module in model.modules():
        if isinstance(module, BiMapLayer):
            module.W._on_stiefel = True

    # Create optimizer
    optimizer = StiefelSGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc: float = 0.0

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume and os.path.exists(".\experiments\checkpoints\spd\spdnet_geom.pt"):
        model, optimizer, start_epoch, best_val_acc = load_resume_checkpoint(
            "experiments/checkpoints/spd/spdnet_geom_latest.pt", model, optimizer, device
        )
        print(
            f"Entrenamiento reanudado desde epoch {start_epoch}, "
            f"mejor val_acc previa={best_val_acc*100:.2f}%"
        )
    else:
        print("Entrenamiento desde cero.")

    # Load json for metrics
    metrics = load_metrics_json(args.json_metrics)

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Training"):
        train_loss, train_acc = train_step(model, train_loader, optimizer, criterion, device, args.lr)
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

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["epochs"].append(epoch)
        save_metrics_json(args.json_metrics, metrics)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.checkpoint, model, optimizer, epoch, best_val_acc)
            tqdm.write(
                f"modelo guardado en la epoch {epoch} con val acc {val_acc}"
            )

    # Save model
    save_checkpoint("experiments/checkpoints/spd/spdnet_geom_latest.pt", model, optimizer, epoch, val_acc)
    # ----------------------------------------------------------
    # Evaluación final (test)
    # ----------------------------------------------------------
    test_acc = test_step(model, test_loader, device)
    print(f"[TEST] acc={test_acc*100:.2f}%")


if __name__ == "__main__":
    main()

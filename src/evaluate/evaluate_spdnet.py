from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
import os

import matplotlib.pyplot as plt

from src.data.datasets import HDM05SPDDataset
from src.data.data_loader import get_dataloaders
from src.models.spdnet import BiMapLayer, SPDNet, StiefelSGD
from src.training.utils import (
    get_device,
    load_checkpoint,
    set_seed,
    load_metrics_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SPDNetGeomstats con HDM05-SPD"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proj_dim", nargs="+", type=int,
                        default=[70, 50, 30])
    parser.add_argument("--checkpoint", type=str,
                        default="experiments/checkpoints/spd/spdnet_geom.pt")
    parser.add_argument("--json_metrics", type=str,
                        default="experiments/checkpoints/spd/spdnet_metrics.json")
    return parser.parse_args()


def plot_metrics(metrics, save_dir="plots"):
    epochs = metrics["epochs"]

    # Loss plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, metrics["train_acc"], label="Train Acc")
    plt.plot(epochs, metrics["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_plot.png"))
    plt.close()


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

    ds = HDM05SPDDataset()

    seed = args.seed
    batch_size = args.batch_size

    train_loader, _, test_loader = get_dataloaders(
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

    model, optimizer = load_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        device
    )

    test_acc = test_step(model, test_loader, device)
    print(f"[TEST] acc={test_acc*100:.2f}%")

    print("plotting...")

    metrics = load_metrics_json(args.metrics_json)
    plot_metrics(metrics)


if __name__ == "__main__":
    main()

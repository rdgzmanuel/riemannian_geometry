# src/hdm05_grassmann/training/train_baseline.py

from __future__ import annotations

import argparse
import os

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument(
        "--checkpoint", type=str, default="experiments/checkpoints/baseline/mlp.pt"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda el entrenamiento desde un checkpoint si existe",
    )
    return parser.parse_args()


def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def train_epoch(
    model: MLPBaseline,
    dataloader: DataLoader,
    optimizer: torch.optim,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    """
    Train one training epoch for the model
    Args:
    - model (MLPBaseline): model
    - dataloader (Dataloader): train dataloader
    - optimizer (torch.optim)
    - criterion (torch.nn.CrossEntropyLoss)
    - device (torch.device)
    Returns:
    - (float): average loss of the epoch
    """
    model.train()
    total_loss: float = 0.0
    total_samples: int = 0

    for x, y in dataloader:
        x: torch.Tensor = x.to(device)
        y: torch.Tensor = y.to(device)

        optimizer.zero_grad()
        logits: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / max(total_samples, 1)


def val_epoch(
    model: MLPBaseline,
    loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple:
    """
    Perform one validation epoch
    Args:
    - model (MLPBaseline): model
    - dataloader (Dataloader): validation dataloader
    - criterion (torch.nn.CrossEntropyLoss)
    - device (torch.device)
    Returns:
    - tuple with average_loss, accuracy

    """
    model.eval()
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    with torch.no_grad():
        for batch in loader:
            x: torch.Tensor
            y: torch.Tensor
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits: torch.Tensor = model(x)
            loss: torch.Tensor = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

    avg_loss: float = total_loss / max(total_samples, 1)
    acc: float = total_correct / max(total_samples, 1)
    return avg_loss, acc


def eval_epoch(
    checkpoint_path: str,
    model: MLPBaseline,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Perform one evaluation epoch of the best model
    Args:
    - checkpoint_path (str): path to the best model
    - model (MLPBaseline): model
    - test_loader (Dataloader): test dataloader
    - device (torch.device)
    Returns:
    - Test accuracy (float)

    """
    print("\nCargando mejor checkpoint para evaluación final...")
    data: dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()

    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for x, y in test_loader:
            x: torch.Tensor = x.to(device)
            y: torch.Tensor = y.to(device)

            logits: torch.Tensor = model(x)
            preds: torch.Tensor = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc: float = correct / max(total, 1)

    print("\n=== Evaluación final ===")
    print(f"test_acc={test_acc*100:.2f}%")

    return test_acc


def load_resume_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load a training checkpoint and restore model weights, optimizer state,
    and metadata required to resume training.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file to load.
    model : nn.Module
        Model instance where the checkpoint weights will be loaded.
    optimizer : optim.Optimizer
        Optimizer whose state will be restored from the checkpoint.
    device : torch.device
        Device where the model and optimizer should be loaded.

    Returns
    -------
    model : nn.Module
        The model with restored weights.
    optimizer : optim.Optimizer
        The optimizer with restored internal state (e.g., momentum).
    start_epoch : int
        The next epoch number to continue training from.
        (checkpoint epoch + 1)
    best_val_acc : float
        The best validation accuracy stored inside the checkpoint.

    """

    print(f"Loading checkpoint to resume training: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch: int = ckpt.get("epoch", 0) + 1
    best_val_acc: float = ckpt.get("best_val_acc", 0.0)

    model.to(device)

    return model, optimizer, start_epoch, best_val_acc


def main():
    args = parse_args()
    set_seed(args.seed)
    device: torch.device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset: usamos un único dataset y lo partimos en train/val
    # ------------------------------------------------------------------
    ds = HDM05WindowsDataset()

    seed = args.seed
    batch_size = args.batch_size

    train_loader, val_loader, test_loader = get_dataloaders(
        ds, batch_size=batch_size, seed=seed
    )

    # Para averiguar T y d, cogemos una muestra
    x0, _ = next(iter(train_loader))  # x0: (T, d)
    _, T, d = x0.shape
    num_classes = len(ds.label2idx)

    print(f"MLPBaseline:{T, d} (T, d) input_dim={T * d}, num_classes={num_classes}")

    model: MLPBaseline = MLPBaseline(input_dim=T * d, num_classes=num_classes).to(
        device
    )
    criterion: torch.nn.CrossEntropyLoss = get_classification_loss("ce")
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc: float = 0.0

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume and os.path.exists(args.checkpoint):
        model, optimizer, start_epoch, best_val_acc = load_resume_checkpoint(
            args.checkpoint, model, optimizer, device
        )
        print(
            f"Entrenamiento reanudado desde epoch {start_epoch}, "
            f"mejor val_acc previa={best_val_acc*100:.2f}%"
        )
    else:
        print("Entrenamiento desde cero.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss: float = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

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

    # Evaluar el mejor modelo guardado
    eval_epoch(args.checkpoint, model, test_loader, device)


if __name__ == "__main__":
    main()

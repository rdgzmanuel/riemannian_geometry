# src/hdm05_grassmann/training/utils.py

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Checkpoint:
    epoch: int
    model_state: dict
    optimizer_state: dict
    best_val_acc: float


def save_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    best_val_acc: float,
):
    """
    Guarda un checkpoint incluyendo:
      - epoch
      - estado del modelo
      - estado del optimizer
      - mejor accuracy de validación
      - nombre de la red (nuevo)
    """

    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "checkpoint_acc": best_val_acc,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"Checkpoint guardado en {path} | acc={best_val_acc:.4f})")


def load_checkpoint(path: str, model, optimizer=None):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])
    print(f"Checkpoint cargado desde {path}")
    return data.get("epoch", 0), data.get("best_val_acc", 0.0)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    logits: (B, C), y: (B,)
    """
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    total = y.numel()
    return correct / total

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


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val_acc: float):
    ckpt = Checkpoint(
        epoch=epoch,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        best_val_acc=best_val_acc,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt.__dict__, path)
    print(f"Checkpoint guardado en {path}")


def load_checkpoint(path: str, model, optimizer=None):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])
    print(f"Checkpoint cargado desde {path}")
    return data.get("epoch", 0), data.get("best_val_acc", 0.0)


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

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    logits: (B, C), y: (B,)
    """
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    total = y.numel()
    return correct / total

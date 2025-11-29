# src/hdm05_grassmann/training/utils.py

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set seed
    Args:
    - seed (int): seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get device
    Args: None
    Returns:
    device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Checkpoint:
    epoch: int
    model_state: dict
    optimizer_state: dict
    best_val_acc: float


def save_checkpoint_spdnet(path: str, model, optimizer, epoch: int, best_val_acc: float) -> None:
    """
    Save a given checkpoint
    Args:
    - path (str): path to save the checkpoint
    - model (nn.Module): model to save
    - optimizer (torch.nn.optim): optimizer used
    - epoch (int): epoch
    - best_val_acc (float): best val accuracy up to the epoch
    """
    ckpt = Checkpoint(
        epoch=epoch,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        best_val_acc=best_val_acc,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt.__dict__, path)
    # print(f"Checkpoint guardado en {path}")

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


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Load a given checkpoint
    Args:
    - path (str): path where the checkpoint is
    - model (nn.Module): model to load
    - optimizer (torch.nn.optim): optimizer used
    - device (torch.device): device used
    """
    
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_state"])
    if optimizer is not None and "optimizer_state" in data:
        optimizer.load_state_dict(data["optimizer_state"])
    print(f"Checkpoint cargado desde {path}")

    model.to(device)

    return model, optimizer


def load_resume_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load a training checkpoint and restore model weights, optimizer state,
    and metadata required to resume training.

    Args:
    - checkpoint_path (str): Path to the checkpoint file to load.
    - model (nn.Module): Model instance where the checkpoint weights will be loaded.
    - optimizer (optim.Optimizer): Optimizer whose state will be restored from the checkpoint.
    - device (torch.device): Device where the model and optimizer should be loaded.

    Returns:
    - model (nn.Module): Model instance where the checkpoint weights will be loaded.
    - optimizer (optim.Optimizer): Optimizer whose state will be restored from the checkpoint.
    - epoch (int): epoch
    - best_val_acc (float): best val accuracy up to the epoch

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
    Get accuracy form the logits
    Args:
    - logits (torch.Tensor): logits tensor (B, C)
    - y (torch.Tensor): correct predictions (B, C)
    Returns:
        accuracy (float)
    """
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    total = y.numel()
    return correct / total


def load_metrics_json(path: str) -> dict:
    """
    Load the metrics of a json
    
    Args:
    - path to the metrics json
    Returns:
     dict with information in the json
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epochs": []
        }


def save_metrics_json(path: str, metrics: dict) -> None:
    """
    Save the metrics as a json
    
    Args:
    - path (str): path to save the metrics json
    - metrics (dict): metrics to save
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
def plot_metrics_history(train_losses, val_losses, train_acc, val_acc):
    # --------------------------------------------------------
    # 1) FIGURA DE LOSS
    # --------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss.pdf", format="pdf")
    plt.close()

    # --------------------------------------------------------
    # 2) FIGURA DE ACCURACY
    # --------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy.pdf", format="pdf")
    plt.close()

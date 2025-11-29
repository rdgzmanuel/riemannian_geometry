# src/hdm05_grassmann/training/eval.py

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion):
    """
    Perform one validation epoch of the best model
    Args:
    - model (nn.Module): model
    - loader (Dataloader): test dataloader
    - device (torch.device)
    Returns:
    - Test accuracy (float), test loss (float)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc

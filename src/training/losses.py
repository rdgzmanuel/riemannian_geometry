# src/hdm05_grassmann/training/losses.py

import torch.nn as nn


def get_classification_loss(name: str = "ce"):
    name = name.lower()
    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss desconocida: {name}")

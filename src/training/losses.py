import torch.nn as nn


def get_classification_loss(name: str = "ce"):
    """
    Get classification loss (Cross Entropy loss)
    """
    name = name.lower()
    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss desconocida: {name}")

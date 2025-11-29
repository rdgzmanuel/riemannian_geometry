# src/hdm05_grassmann/models/baselines.py

from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Simple MLP to classify Windows (T, d)

    """

    def __init__(
        self,
        input_dim: int,  # T * d
        num_classes: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        """
        Args:
        - input_dim (int): Txd
        - num_classes (int)
        - hidden_dims (list)
        - dropout (float)
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        dims = [input_dim] + hidden_dims
        layers = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=False):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass
        Args:
        - x (torch.Tensor): (B, T, d)
        Returns:
        - output (torch.Tensor): (B, C)
        """
        B, T, d = x.shape
        x = x.reshape(B, T * d)  # (B, T*d)
        h = self.mlp(x)  # (B, hidden)
        logits = self.classifier(h)  # (B, C)
        return logits

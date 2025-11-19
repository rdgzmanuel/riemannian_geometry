from __future__ import annotations

import torch
import torch.nn as nn

from src.manifolds.spd_ops import re_eig, log_map_spd


# -------------------------------------------------
# 1. Capa BiMap:  X -> W^T X W
# -------------------------------------------------
class BiMap(nn.Module):
    """
    BiMap layer:
      X ∈ R^{B×d_in×d_in}  →  Y ∈ R^{B×d_out×d_out}
      Y_b = W^T X_b W
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Inicialización pequeña para no salirnos mucho de la identidad
        self.W = nn.Parameter(torch.randn(d_in, d_out) * 0.1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, d_in, d_in)
        return: (B, d_out, d_out)
        """
        # XW: (B, d_in, d_out)
        W = self.W / (torch.norm(self.W, dim=0, keepdim=True) + 1e-8)
        XW = torch.einsum("bij,jk->bik", X, W)
        # XW = torch.einsum("bij,jk->bik", X, self.W)
        # W^T (XW): (B, d_out, d_out)
        Y = torch.einsum("ij,bjk->bik", W.transpose(0, 1), XW)
        # Y = torch.einsum("ij,bjk->bik", self.W.transpose(0, 1), XW)
        return Y


# -------------------------------------------------
# 2. Capa ReEig (clamp de autovalores)
# -------------------------------------------------
class ReEig(nn.Module):
    """
    ReEig:
      Proyecta de nuevo a SPD clampando autovalores a eps.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, d, d) simétrica (aprox.)
        """
        return re_eig(X, eps=self.eps)


# -------------------------------------------------
# 3. Modelo SPDNet puro PyTorch (sin geomstats)
# -------------------------------------------------
class SPDNet(nn.Module):
    """
    SPDNet con logaritmo matricial vía eigen-descomposición (PyTorch puro).

    Arquitectura:
      X SPD (d_in×d_in)
        -> BiMap(d_in -> proj_dim)
        -> ReEig
        -> Log (log_eig, métrica "log-euclidean"/"affine")
        -> BiMap(proj_dim -> proj_dim)
        -> ReEig
        -> Log
        -> Flatten
        -> Linear -> num_classes
    """

    def __init__(
        self,
        d_in: int,
        proj_dim: int = 32,
        num_classes: int = 130,
        metric: str = "log-euclidean",
    ):
        super().__init__()
        self.d_in = d_in
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.metric = metric

        # Bloque 1
        self.bimap1 = BiMap(d_in, proj_dim)
        self.reig1 = ReEig()

        # Bloque 2
        self.bimap2 = BiMap(proj_dim, proj_dim)
        self.reig2 = ReEig()

        # Clasificador Euclídeo
        self.fc = nn.Linear(proj_dim * proj_dim, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, d_in, d_in) SPD
        """
        # Bloque 1
        X = self.bimap1(X)           # (B, proj_dim, proj_dim)
        if not torch.isfinite(X).all():
            print("NaN in BiMap1"); raise SystemExit
        X = self.reig1(X)            # asegurar SPD
        if not torch.isfinite(X).all():
            print("NaN/Inf after block 1"); raise SystemExit
        X = log_map_spd(X, metric=self.metric)  # (B, proj_dim, proj_dim)

        # Bloque 2
        X = self.bimap2(X)
        X = self.reig2(X)
        if not torch.isfinite(X).all():
            print("NaN/Inf after block 1"); raise SystemExit
        X = log_map_spd(X, metric=self.metric)

        # Flatten + clasificación
        B, d, _ = X.shape
        X_flat = X.reshape(B, d * d)
        out = self.fc(X_flat)
        return out

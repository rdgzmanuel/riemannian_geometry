from __future__ import annotations

import torch
import torch.nn as nn

from src.manifolds.spd_ops import re_eig, log_map_spd


class BiReBlock(nn.Module): 
    """
    BiMap layer: 
    X ∈ R^{B×d_in×d_in} → Y ∈ R^{B×d_out×d_out} 
    Y_b = W X_b W^T
    """ 
    def __init__(
        self,
        d_in: int,
        d_out: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.eps = eps
        self.W = nn.Parameter(torch.randn(d_out, d_in))
        self.reset_parameters()

    def reset_parameters(self):
        # Inicialización semi-ortogonal (filas ortonormales)
        with torch.no_grad():
            # QR sobre Wᵀ: (d_in, d_out) -> q: (d_in, d_out)
            q, _ = torch.linalg.qr(self.W.t())  # columnas ortonormales en q
            self.W.copy_(q.t())                 # filas ortonormales en W

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        X: (B, d_in, d_in) return: (B, d_out, d_out)
        """
        # (paper huang andd gool)
        # -------------------------------------------------
        # 1. Capa BiMap: X -> W X W^T
        # -------------------------------------------------
        # Proyección a Stiefel en cada forward (aprox Riemanniano)
        W = self.W
        q, _ = torch.linalg.qr(W.t())    # (d_in, d_out)
        W_st = q.t()                     # (d_out, d_in)

        # WX: (B, d_in, d_out)
        WX = torch.matmul(W_st, X)

        # Y = WWXᵀ: (B, d_out, d_out)
        Y = torch.matmul(WX, W.t())

        if not torch.isfinite(Y).all():
            print("NaN in BiMap")
            raise SystemExit

        # -------------------------------------------------
        # 2. Capa ReEig (clamp de autovalores)
        # -------------------------------------------------
        return re_eig(Y, eps=self.eps)


class SPDNet(nn.Module):
    """
    SPDNet con logaritmo matricial vía eigen-descomposición (PyTorch puro).

    Arquitectura:
      X SPD (d_in×d_in)
        -> BiMap(d_in -> proj_dim)
        -> ReEig
        -> BiMap(d_in -> proj_dim)
        -> ReEig
        -> BiMap(d_in -> proj_dim)
        -> ReEig
        -> Log
        -> Flatten
        -> Linear -> num_classes
    """

    def __init__(
        self,
        d_in: int,
        proj_dim: int = [70, 50, 30],
        num_classes: int = 70,
        metric: str = "log-euclidean",
    ):
        super().__init__()
        self.d_in = d_in
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.metric = metric

        # layers
        self.birelayer1 = BiReBlock(d_in, proj_dim[0])
        self.birelayer2 = BiReBlock(proj_dim[0], proj_dim[1])
        self.birelayer3 = BiReBlock(proj_dim[1], proj_dim[2])
        # Clasificador Euclídeo
        d = proj_dim[-1]
        self.fc = nn.Linear(d * d, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, d_in, d_in) SPD
        """

        # 3 bloques de BiRe (BiMap y Re)
        X = self.birelayer1(X)
        X = self.birelayer2(X)
        X = self.birelayer3(X)

        X = log_map_spd(X, metric=self.metric)

        # Flatten + clasificación
        B, d, _ = X.shape
        X_flat = X.reshape(B, d * d)
        out = self.fc(X_flat)
        return out

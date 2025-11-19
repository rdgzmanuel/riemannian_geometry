# src/hdm05_grassmann/models/grnet.py

from __future__ import annotations

import torch
import torch.nn as nn
from src.manifolds.grassman_geomstats import get_grassmann_geomstats


class FRMapGeomstats(nn.Module):
    """
    Capa FRMap para subespacios en Grassmann usando Geomstats (Stiefel).

    U ∈ R^{d×p_in} (ortonormal)  →  U_out ∈ R^{d×p_out} (ortonormal)

        Y = U W            (mapa lineal en el subespacio)
        U_out = Proj(Y)    (proyección al manifold Stiefel(d, p_out))

    Donde Proj es la proyección de Geomstats (SVD → columnas ortonormales).
    """

    def __init__(self, d: int, p_in: int, p_out: int, bias: bool = False):
        super().__init__()
        self.d = d
        self.p_in = p_in
        self.p_out = p_out

        self.weight = nn.Parameter(torch.randn(p_in, p_out) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1, 1, p_out)) if bias else None

        # manifold geométrico con Geomstats
        self.manifold = get_grassmann_geomstats(d, p_out)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        U: (B, d, p_in)
        """
        B, d, p_in = U.shape
        assert d == self.d and p_in == self.p_in

        # Mapa lineal en el subespacio
        Y = U @ self.weight  # (B, d, p_out)
        if self.bias is not None:
            Y = Y + self.bias  # broadcast (1,1,p_out)

        # Mapa lineal en el subespacio
        Y = U @ self.weight  # (B, d, p_out)
        if self.bias is not None:
            Y = Y + self.bias  # broadcast (1,1,p_out)

        # (Opcional) puedes estabilizar un poco acotando valores extremos:
        # Y = torch.clamp(Y, min=-1e3, max=1e3)

        # Proyección al manifold Stiefel(d, p_out) mediante QR batched
        # torch.linalg.qr soporta tensores batched: (B, d, p_out) → Q: (B, d, p_out)
        Q, _ = torch.linalg.qr(Y, mode="reduced")

        # Q tiene columnas ortonormales → base de subespacio en Stiefel/Grassmann
        U_out = Q  # (B, d, p_out)

        return U_out


class GrassmannFlatten(nn.Module):
    """
    Aplasta U ∈ ℝ^{B×d×p} → ℝ^{B×(d·p)} para pasarlo a MLP euclídeo.
    """

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        B, d, p = U.shape
        return U.reshape(B, d * p)


class GrassmannNetGeomstats(nn.Module):
    """
    GrNet usando Geomstats + representación Stiefel para Grassmann.

    Pipeline:
      Input:  U ∈ ℝ^{B×d×p_in}  (subespacios)
        - varias capas FRMapGeomstats (d × p_in → d × p_out)
        - flatten
        - MLP euclídeo
        - logits de clasificación
    """

    def __init__(
        self,
        d: int,
        p_in: int,
        num_classes: int,
        fr_layers: list[int] | None = None,  # lista de p_out por capa
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if fr_layers is None:
            fr_layers = [32, 64, 32]

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # 1) Capas FRMap
        frmaps = []
        p_prev = p_in
        for p_out in fr_layers:
            frmaps.append(FRMapGeomstats(d, p_prev, p_out))
            p_prev = p_out

        self.frmaps = nn.ModuleList(frmaps)

        # 2) Flatten + MLP
        self.flatten = GrassmannFlatten()

        input_fc_dim = d * p_prev
        dims = [input_fc_dim] + hidden_dims
        mlp_layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=False):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp_layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        U: (B, d, p_in)
        """
        for fr in self.frmaps:
            U = fr(U)  # (B, d, p')

        h = self.flatten(U)  # (B, d*p_last)
        h = self.mlp(h)  # (B, hidden)
        logits = self.classifier(h)  # (B, C)
        return logits

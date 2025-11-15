# src/hdm05_grassmann/models/grgcn.py
from __future__ import annotations

import torch
import torch.nn as nn
from src.manifolds.grassman_geomstats import get_grassmann_geomstats


class GrGCNLayerGeomstats(nn.Module):
    """
    Capa tipo GCN sobre nodos en Grassmann, usando representación Stiefel.

    U: (B, N, d, p_in)  nodos = subespacios
    A: (B, N, N) o (N, N)  matriz de adyacencia

    Operación:
      - Mensaje propio:   U_i W_self
      - Mensaje vecinos:  sum_j A_ij U_j W_neigh
      - Y_i = self + vecinos
      - U'_i = Proj(Y_i)  (proyección a Stiefel(d, p_out) con Geomstats)
    """

    def __init__(self, d: int, p_in: int, p_out: int, bias: bool = False):
        super().__init__()
        self.d = d
        self.p_in = p_in
        self.p_out = p_out

        self.weight_self = nn.Parameter(torch.randn(p_in, p_out) * 0.01)
        self.weight_neigh = nn.Parameter(torch.randn(p_in, p_out) * 0.01)

        self.bias = nn.Parameter(torch.zeros(1, 1, 1, p_out)) if bias else None

        self.manifold = get_grassmann_geomstats(d, p_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, U: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        U: (B, N, d, p_in)
        A: (B, N, N) o (N, N)
        """
        B, N, d, p_in = U.shape
        assert d == self.d and p_in == self.p_in

        # Mensajes propio y vecinos (en el subespacio)
        M_self = U @ self.weight_self  # (B, N, d, p_out)
        M_neigh = U @ self.weight_neigh  # (B, N, d, p_out)

        if A.dim() == 2:
            A = A.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # Agregamos vecinos: A @ M_neigh (a través de un flatten)
        B, N, d, p_out = M_neigh.shape
        M_neigh_flat = M_neigh.reshape(B, N, d * p_out)  # (B, N, d*p_out)
        agg_flat = A @ M_neigh_flat  # (B, N, d*p_out)
        M_agg = agg_flat.reshape(B, N, d, p_out)  # (B, N, d, p_out)

        Y = M_self + M_agg  # (B, N, d, p_out)
        if self.bias is not None:
            Y = Y + self.bias

        # No linealidad euclídea en el ambiente (opcional)
        Y = self.act(Y)

        # Proyección al manifold por nodo:
        # apilamos (B*N, d, p_out), proyectamos, des-apilamos.
        Y_flat = Y.reshape(B * N, d, p_out)  # (B*N, d, p_out)
        U_proj_flat = self.manifold.projection(Y_flat)  # (B*N, d, p_out)
        U_out = U_proj_flat.reshape(B, N, d, p_out)

        return U_out


class GrassmannNodeMeanPool(nn.Module):
    """
    Pooling simple de nodos:
      - media euclídea en el ambiente,
      - proyección a Stiefel(d, p) con Geomstats.
    """

    def __init__(self, d: int, p: int):
        super().__init__()
        self.manifold = get_grassmann_geomstats(d, p)

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        """
        U: (B, N, d, p)
        """
        B, N, d, p = U.shape
        U_mean = U.mean(dim=1)  # (B, d, p)
        U_proj = self.manifold.projection(U_mean)
        return U_proj  # (B, d, p)


class GrGCNPlusPlusNetGeomstats(nn.Module):
    """
    Versión GrGCN++ usando Geomstats + representación Stiefel.

    Input: U (B, N, d, p_in), A (B, N, N) o (N, N)
      → varias capas GrGCNLayerGeomstats
      → pooling de nodos (GrassmannNodeMeanPool)
      → flatten
      → MLP euclídeo
      → logits
    """

    def __init__(
        self,
        d: int,
        p_in: int,
        num_classes: int,
        gcn_layers: list[int] | None = None,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
    ):
        super().__init__()

        if gcn_layers is None:
            gcn_layers = [16, 8]

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        p_prev = p_in
        for p_out in gcn_layers:
            layers.append(GrGCNLayerGeomstats(d, p_prev, p_out))
            p_prev = p_out

        self.gcn_layers = nn.ModuleList(layers)
        self.pool = GrassmannNodeMeanPool(d, p_prev)

        # Flatten + MLP
        input_fc_dim = d * p_prev
        dims = [input_fc_dim] + hidden_dims
        mlp_layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=False):
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            mlp_layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp_layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, U: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        U: (B, N, d, p_in)
        A: (B, N, N) o (N, N)
        """
        for layer in self.gcn_layers:
            U = layer(U, A)  # (B, N, d, p')

        U_pool = self.pool(U)  # (B, d, p_last)
        B, d, p_last = U_pool.shape
        h = U_pool.reshape(B, d * p_last)  # (B, d*p_last)
        h = self.mlp(h)  # (B, hidden)
        logits = self.classifier(h)  # (B, C)
        return logits

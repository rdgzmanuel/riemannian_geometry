# # src/hdm05_grassmann/manifolds/ops.py

# import geoopt
# import torch

# # ------------------------------------------------------------
# # GEOMETRIA BÁSICA EN GRASSMANN
# # ------------------------------------------------------------


# def proj_to_tangent(U: torch.Tensor, Xi: torch.Tensor) -> torch.Tensor:
#     """
#     Proyección de Xi al espacio tangente en U.
#     Tangente:  T_U Gr = { Z : Uᵀ Z = 0 }
#     """
#     return Xi - U @ (U.transpose(-1, -2) @ Xi)


# def exp_map(U: torch.Tensor, Xi: torch.Tensor, manifold: geoopt.manifolds):
#     """
#     Mapa exponencial en Grassmann.
#     Lleva un vector tangente Xi a un punto del manifold.
#     """
#     return manifold.expmap(Xi, U)


# def log_map(U: torch.Tensor, V: torch.Tensor, manifold: geoopt.manifolds):
#     """
#     Mapa logarítmico en Grassmann:
#         devuelve el vector tangente que lleva U a V.
#     """
#     return manifold.logmap(V, U)


# def dist(U: torch.Tensor, V: torch.Tensor, manifold: geoopt.manifolds):
#     """
#     Distancia geodésica en Grassmann.
#     """
#     return manifold.dist(U, V)


# # ------------------------------------------------------------
# # NORMALIZACIÓN
# # ------------------------------------------------------------


# def reorthonormalize(U: torch.Tensor) -> torch.Tensor:
#     """
#     Re-ortonormaliza U usando QR.
#     """
#     Q, _ = torch.linalg.qr(U)
#     return Q


# # ------------------------------------------------------------
# # MEDIA GEOMÉTRICA (opcional)
# # ------------------------------------------------------------


# def mean_subspaces(U_list, manifold: geoopt.manifolds, max_iter=10):
#     """
#     Media geométrica simple (Karcher mean) para un conjunto de subespacios.
#     Muy útil para pooling Riemanniano.

#     U_list: lista de tensores U_i (d,p)
#     """
#     U = U_list[0].clone()

#     for _ in range(max_iter):
#         # vector medio en el espacio tangente
#         Xi = sum([log_map(U, Ui, manifold) for Ui in U_list]) / len(U_list)
#         # mover U en esa dirección
#         U = exp_map(U, Xi, manifold)

#     return U


# # ------------------------------------------------------------
# # OPERACIONES POR LOTE
# # ------------------------------------------------------------


# def batch_dist(U: torch.Tensor, V: torch.Tensor, manifold):
#     """
#     Distancia batched:
#         U: (B, d, p)
#         V: (B, d, p)
#     """
#     B = U.size(0)
#     return torch.stack([manifold.dist(U[i], V[i]) for i in range(B)], dim=0)


# def batch_reorth(U: torch.Tensor):
#     """
#     QR batch: U shape (B, d, p)
#     """
#     B = U.size(0)
#     return torch.stack([reorthonormalize(U[i]) for i in range(B)], dim=0)


# src/hdm05_grassmann/manifolds/grassmann_geomstats.py

import geomstats.backend as gs
from geomstats.geometry.grassmannian import Grassmannian

gs.set_default_backend("pytorch")


class GrassmannManifold:
    def __init__(self, d, p):
        self.d = d
        self.p = p
        self.G = Grassmannian(n=d, k=p)

    def proj(self, U):
        """Proyección al manifold (SVD o QR)."""
        return self.G.projection(U)

    def to_tangent(self, X, U):
        return self.G.to_tangent(X, U)  # proyección tangente

    def exp(self, Xi, U):
        return self.G.metric.exp(Xi, U)  # expmap

    def log(self, V, U):
        return self.G.metric.log(V, U)

    def dist(self, U, V):
        return self.G.metric.dist(U, V)

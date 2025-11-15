# src/hdm05_grassmann/manifolds/grassmann_geoopt.py

from __future__ import annotations

from dataclasses import dataclass

import geoopt
import torch


@dataclass
class GrassmannManifold:
    """
    Wrapper ligero sobre el Stiefel de geoopt, que usamos como modelo
    del Grassmann: puntos = matrices ortonormales U ∈ R^{d×p}.

    d: dimensión ambiente
    p: dimensión del subespacio
    """

    d: int
    p: int

    def __post_init__(self):
        # Geoopt NO tiene Grassmann, usamos Stiefel (X^T X = I).
        # La geometría que implementa (expmap, logmap, dist) es la correcta
        # para matrices ortonormales, que es justo nuestra representación.
        self.manifold = geoopt.manifolds.Stiefel()  # sin args: usa últimas 2 dims

    def check_point(self, U: torch.Tensor):
        """
        Comprobación rápida de shape: (..., d, p).
        """
        if U.shape[-2:] != (self.d, self.p):
            raise ValueError(
                f"Esperaba punto en Grassmann con shape (*,{self.d},{self.p}), "
                f"recibido {tuple(U.shape)}"
            )


# helper para crear el wrapper
_grassmann_cache: dict[tuple[int, int], GrassmannManifold] = {}


def get_grassmann(d: int, p: int) -> GrassmannManifold:
    key = (d, p)
    if key not in _grassmann_cache:
        _grassmann_cache[key] = GrassmannManifold(d, p)
    return _grassmann_cache[key]

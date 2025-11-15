from __future__ import annotations

from geomstats.geometry.stiefel import Stiefel

# Usar PyTorch como backend (muy importante)
# gs.set_default_backend("pytorch")


class GrassmannGeomstats:
    """
    Wrapper de Stiefel(n, p) de Geomstats, que usamos como
    representación de subespacios en la Grassmann Gr(n, p).

    Cada punto se representa como U ∈ R^{n×p} con columnas ortonormales.
    """

    def __init__(self, n: int, p: int):
        self.n = n
        self.p = p
        self.space = Stiefel(n=n, p=p)
        self.metric = self.space.metric  # StiefelCanonicalMetric

    def projection(self, U):
        """Proyección al manifold (tipo QR/SVD).  shape (..., n, p)."""
        return self.space.projection(U)

    def to_tangent(self, V, U):
        """Proyección de V al espacio tangente en U."""
        return self.space.to_tangent(V, U)

    def retraction(self, Xi, U):
        """Retraction tipo QR: base_point + Xi → manifold."""
        return self.metric.retraction(Xi, U)

    # Por si luego quieres usar exp/log/dist:
    def exp(self, Xi, U):
        return self.metric.exp(Xi, U)

    def log(self, V, U):
        return self.metric.log(V, U)

    def dist(self, U, V):
        return self.metric.dist(U, V)


# Pequeña caché para no crear mil veces el mismo manifold
_cache: dict[tuple[int, int], GrassmannGeomstats] = {}


def get_grassmann_geomstats(n: int, p: int) -> GrassmannGeomstats:
    key = (n, p)
    if key not in _cache:
        _cache[key] = GrassmannGeomstats(n, p)
    return _cache[key]

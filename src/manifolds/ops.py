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

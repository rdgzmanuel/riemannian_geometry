from __future__ import annotations
import torch


def symmetrize(X: torch.Tensor) -> torch.Tensor:
    """Ensure X is symmetric: X = (X + Xᵀ)/2."""
    return 0.5 * (X + X.transpose(-1, -2))


def re_eig(X: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Projection to SPD using eigen-decomposition.
    MUCH more stable than the previous version.
    """
    # Asegurar simetría
    X = symmetrize(X)

    # eigh: para matrices simétricas (SPD), más estable que svd
    evals, evecs = torch.linalg.eigh(X)  # (..., d), (..., d, d)

    # Clamp fuerte a valores positivos
    evals_clamped = torch.clamp(evals, min=eps)

    # Reconstrucción SPD: U diag(S) U^T
    X_clamped = evecs @ torch.diag_embed(evals_clamped) @ evecs.transpose(-1, -2)
    return symmetrize(X_clamped)


def log_eig(X: torch.Tensor) -> torch.Tensor:
    """
    Logarithm of SPD matrix using eigen-decomposition (stable).
    """
    X = symmetrize(X)
    evals, evecs = torch.linalg.eigh(X)
    log_evals = torch.log(evals)
    X_log = evecs @ torch.diag_embed(log_evals) @ evecs.transpose(-1, -2)
    return symmetrize(X_log)


def log_map_spd(
    X: torch.Tensor,
    metric: str = "log-euclidean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Log-map in SPD manifold at identity.
    Only Log-Euclidean is implemented here.
    """
    metric = metric.lower()

    if metric in ("log-euclidean", "log_euclidean", "log"):
        return log_eig(X, eps=eps)
    else:
        raise ValueError(f"Unknown SPD metric: {metric}")

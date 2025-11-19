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
    U, S, Vt = torch.linalg.svd(X)

    # Clamp fuerte a valores positivos
    S_clamped = torch.clamp(S, min=eps)

    # Reconstrucción SPD: U diag(S) U^T
    X_spd = U @ torch.diag_embed(S_clamped) @ U.transpose(-1, -2)

    # Re-simetrizar
    return symmetrize(X_spd)


def log_eig(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Logarithm of SPD matrix using eigen-decomposition (stable).
    """
    X = symmetrize(X)
    U, S, _ = torch.linalg.svd(X)

    S = torch.clamp(S, min=eps)
    logS = torch.log(S)

    X_log = U @ torch.diag_embed(logS) @ U.transpose(-1, -2)
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

    elif metric == "affine":
        # Using log_eig as SPDNet original does for base_point=I
        return log_eig(X, eps=eps)

    else:
        raise ValueError(f"Unknown SPD metric: {metric}")

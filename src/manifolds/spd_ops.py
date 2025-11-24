from __future__ import annotations
import torch


def compute_P_matrix(
    eigenvalues: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Compute P matrix (Eq. 14): P(i,j) = 1/(sigma_i - sigma_j) for i != j
    Args:
        eigenvalues: (batch_size, n)
    Returns:
        P: (batch_size, n, n)
    """
    # Add small epsilon to avoid division by zero
    # eps = 1e-8

    # # Compute pairwise differences
    # diff = eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)

    # # Avoid division by zero on diagonal
    # mask = torch.eye(eigenvalues.size(-1), device=eigenvalues.device).bool()
    # diff = diff + mask.float() * eps

    # P = 1.0 / diff

    # # Set diagonal to zero
    # P = P * (~mask).float()

    # Compute pairwise differences
    diff = eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2)

    # CORREGIDO: Usar una función suavizada en lugar de 1/x directo
    # P(i,j) = tanh(diff / eps) / diff  cuando diff != 0
    mask_diag = torch.eye(eigenvalues.size(-1),
                          device=eigenvalues.device).unsqueeze(0)

    # Evitar división por cero
    # diff_safe = diff + mask_diag.float() * eps
    mask_small = diff.abs() < eps
    diff_safe = diff.clone()
    diff_safe[mask_small] = eps

    # Suavizado con tanh para diferencias pequeñas
    P = 1 / diff_safe

    # Set diagonal to zero
    P = P * (~mask_small)
    P = P * (1.0 - mask_diag)

    # Clamp para evitar valores extremos
    P = torch.clamp(P, min=-eps, max=eps)

    return P


def  retraction_stiefel(W, gradient, lr):
    """
    Retraction operation to project back onto Stiefel manifold (Eq. 8)
    Args:
        W: Current weight on Stiefel manifold (n, m)
        gradient: Riemannian gradient (n, m)
        lr: Learning rate
    Returns:
        Updated weight on Stiefel manifold
    """
    # Update in tangent space
    W_updated = W - lr * gradient

    # QR decomposition for retraction
    Q, _ = torch.linalg.qr(W_updated.t())

    return Q.t()


def symmetrize(X: torch.Tensor) -> torch.Tensor:
    """Ensure X is symmetric: X = (X + Xᵀ)/2."""
    return 0.5 * (X + X.transpose(-1, -2))

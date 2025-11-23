from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.manifolds.spd_ops import compute_P_matrix, retraction_stiefel


def stiefel_init_param(W: torch.nn.Parameter) -> None:
    # W: (d_out, d_in)
    d_out, d_in = W.shape
    if d_out > d_in:
        A = torch.randn(d_out, d_in, device=W.device, dtype=W.dtype)
        q, r = torch.linalg.qr(A.T)
        q = q * torch.sign(torch.diag(r).clamp(min=1e-6))[:, None]
        W.data.copy_(q.t()[:d_out, :d_in])
    else:
        A = torch.randn(d_in, d_out, device=W.device, dtype=W.dtype)
        q, r = torch.linalg.qr(A)
        diag = torch.sign(torch.diag(r))
        q = q * diag[None, :]
        W.data.copy_(q.t())


class BiMapFunction(Function):
    """
    BiMap: X_k = W_k^T X_{k-1} W_k
    """
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        X: (B, d_in, d_in) return: (B, d_out, d_out)
        """
        # (paper huang andd gool)
        # Capa BiMap: X -> W X W^T

        # Context
        ctx.save_for_backward(X, W)

        # Proyección a Stiefel en cada forward (aprox Riemanniano)
        q, _ = torch.linalg.qr(W.t())    # (d_in, d_out)
        W_st = q.t()                     # (d_out, d_in)

        # WX: (B, d_in, d_out)
        WX = torch.matmul(W_st, X)

        # Y = WWXᵀ: (B, d_out, d_out)
        Y = torch.matmul(WX, W_st.t())

        if not torch.isfinite(Y).all():
            print("NaN/Inf en BiMap (Y antes de ReEig)")
            # print("  WX stats:",
            #       "min", WX.min().item(),
            #       "max", WX.max().item())
            # print("  W_st stats:",
            #       "min", W_st.min().item(),
            #       "max", W_st.max().item())
            # print("  W stats:",
            #       "min", self.W.min().item(),
            #       "max", self.W.max().item())
            raise SystemExit

        return Y

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.tensor,
    ):
        """
        Args:
            grad_output: Gradient from upper layer dL/dX_k (batch_size, m, m)
        Returns:
            grad_X: Gradient w.r.t. X (batch_size, n, n)
            grad_W: Riemannian gradient w.r.t. W (n, m)
        """
        X, W = ctx.saved_tensors

        # Gradient w.r.t. X_{k-1} using chain rule
        # dL/dX_{k-1} = W (dL/dX_k) W^T
        grad_X = torch.matmul(torch.matmul(W.t(), grad_output), W)

        # Euclidean gradient w.r.t. W (Eq. 9)
        # ∇L^(k)_{W_k} = 2 (dL/dX_k) W_k X_{k-1}
        mult = torch.matmul(W, X)
        euclidean_grad = 2 * torch.matmul(grad_output, mult)

        # Convert to Riemannian gradient (Eq. 7)
        # ∇̃L^(k)_{W_k} = ∇L^(k)_{W_k} - ∇L^(k)_{W_k} (W_k)^T W_k
        grad_W = euclidean_grad - torch.matmul(
            torch.matmul(euclidean_grad, W.t()), W
        )

        return grad_X, grad_W


class ReEigFunction(Function):
    """
    ReEig layer: Rectifies eigenvalues to be >= epsilon
    """
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        epsilon: float = 1e-4,
    ):
        """
        Args:
            X: Input SPD matrix (batch_size, n, n)
            epsilon: Threshold for rectification
        Returns:
            Output SPD matrix with rectified eigenvalues
        """
        # Para evitar errores
        # epsilon = 1e-4
        X = X + epsilon * torch.eye(X.size(-1), device=X.device)

        if not torch.isfinite(X).all():
            print("NaN o Inf en X en ReEig")
            print(X)
            raise SystemExit

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(X)

        eigenvalues_rectified = torch.clamp(eigenvalues, min=epsilon, max=1e6)

        ctx.save_for_backward(eigenvectors, eigenvalues, eigenvalues_rectified)
        ctx.epsilon = epsilon

        # Reconstruct: X_k = U max(I*epsilon, Sigma) U^T
        output = torch.matmul(
            torch.matmul(eigenvectors,
                         torch.diag_embed(eigenvalues_rectified)),
            eigenvectors.transpose(-2, -1)
        )
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ):
        """
        Args:
            grad_output: Gradient from upper layer dL/dX_k
        Returns:
            grad_X: Gradient w.r.t. input X_{k-1}
        """
        U, Sigma, Sigma_rect = ctx.saved_tensors
        epsilon = ctx.epsilon

        # Make gradient symmetric
        grad_sym = 0.5 * (grad_output + grad_output.transpose(-2, -1))

        # Compute Q (Eq. 18): gradient of max(epsilon, sigma)
        Q = (Sigma > epsilon).float()
        Q_diag = torch.diag_embed(Q)

        # Compute dL/dU (Eq. 16)
        # dL/dU = 2 * (dL/dX_k)_sym * U * max(I*epsilon, Sigma)
        dL_dU = 2 * torch.matmul(
            torch.matmul(grad_sym, U),
            torch.diag_embed(Sigma_rect)
        )

        # Compute dL/dSigma (Eq. 17)
        # dL/dSigma = Q * U^T * (dL/dX_k)_sym * U
        dL_dSigma = Q_diag * torch.matmul(
            torch.matmul(U.transpose(-2, -1), grad_sym), U
        )

        # Extract diagonal
        dL_dSigma_diag = torch.diagonal(dL_dSigma, dim1=-2, dim2=-1)

        # Compute P matrix (Eq. 14) for eigenvalue differences
        P = compute_P_matrix(Sigma)

        # Compute U^T * dL/dU
        UT_dL_dU = torch.matmul(U.transpose(-2, -1), dL_dU)

        # Symmetrize and apply Hadamard product with P^T
        UT_dL_dU_sym = 0.5 * (UT_dL_dU + UT_dL_dU.transpose(-2, -1))
        hadamard_term = P.transpose(-2, -1) * UT_dL_dU_sym

        # Final gradient (Eq. 15)
        grad_X = 2 * torch.matmul(torch.matmul(U, hadamard_term), U.transpose(-2, -1))
        grad_X += torch.matmul(
            torch.matmul(U, torch.diag_embed(dL_dSigma_diag)),
            U.transpose(-2, -1)
        )

        return grad_X, None


class LogEigFunction(Function):
    """
    LogEig layer: Takes logarithm of eigenvalues
    """
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor
    ):
        """
        Args:
            X: Input SPD matrix (batch_size, n, n)
        Returns:
            Output matrix with log eigenvalues
        """
        # Eigenvalue decomposition
        X = X + 1e-4 * torch.eye(X.size(-1), device=X.device)

        eigenvalues, eigenvectors = torch.linalg.eigh(X)

        eigenvalues_safe = torch.clamp(eigenvalues, min=1e-6)
        # Take logarithm of eigenvalues
        log_eigenvalues = torch.log(eigenvalues_safe)

        ctx.save_for_backward(eigenvectors, eigenvalues_safe)

        # Reconstruct: X_k = U log(Sigma) U^T
        output = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(log_eigenvalues)),
            eigenvectors.transpose(-2, -1)
        )
        # print("LogEig here", output)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ):
        """
        Args:
            grad_output: Gradient from upper layer dL/dX_k
        Returns:
            grad_X: Gradient w.r.t. input X_{k-1}
        """
        U, Sigma = ctx.saved_tensors

        # Make gradient symmetric
        grad_sym = 0.5 * (grad_output + grad_output.transpose(-2, -1))

        # Compute log(Sigma)
        log_Sigma = torch.log(Sigma)

        # Compute dL/dU (Eq. 19)
        # dL/dU = 2 * (dL/dX_k)_sym * U * log(Sigma)
        dL_dU = 2 * torch.matmul(
            torch.matmul(grad_sym, U),
            torch.diag_embed(log_Sigma)
        )

        # Compute dL/dSigma (Eq. 20)
        # dL/dSigma = Sigma^{-1} * U^T * (dL/dX_k)_sym * U
        Sigma_inv = 1.0 / Sigma
        dL_dSigma = torch.diag_embed(Sigma_inv) * torch.matmul(
            torch.matmul(U.transpose(-2, -1), grad_sym), U
        )

        # Extract diagonal
        dL_dSigma_diag = torch.diagonal(dL_dSigma, dim1=-2, dim2=-1)

        # Compute P matrix (Eq. 14)
        P = compute_P_matrix(Sigma)

        # Compute U^T * dL/dU
        UT_dL_dU = torch.matmul(U.transpose(-2, -1), dL_dU)

        # Symmetrize and apply Hadamard product with P^T
        UT_dL_dU_sym = 0.5 * (UT_dL_dU + UT_dL_dU.transpose(-2, -1))
        hadamard_term = P.transpose(-2, -1) * UT_dL_dU_sym

        # Final gradient (Eq. 15)
        grad_X = 2 * torch.matmul(torch.matmul(U, hadamard_term),
                                  U.transpose(-2, -1))
        grad_X += torch.matmul(
            torch.matmul(U, torch.diag_embed(dL_dSigma_diag)),
            U.transpose(-2, -1)
        )

        return grad_X


class BiMapLayer(nn.Module):
    """
    BiMap layer: 
    X ∈ R^{B×d_in×d_in} → Y ∈ R^{B×d_out×d_out} 
    Y_b = W X_b W^T
    """ 
    def __init__(
        self,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W = nn.Parameter(torch.randn(d_out, d_in))

        stiefel_init_param(self.W)

    def forward(
        self,
        X: torch.Tensor,
    ):
        return BiMapFunction.apply(X, self.W)


class ReEigLayer(nn.Module):
    def __init__(
        self,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        X: torch.Tensor,
    ):
        return ReEigFunction.apply(X, self.eps)


class LogEigLayer(nn.Module):
    """LogEig Layer: Takes logarithm of eigenvalues"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        X: torch.Tensor
    ):
        return LogEigFunction.apply(X)


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
    ):
        super().__init__()
        self.d_in = d_in
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        layers = []
        current_dim = self.d_in

        # Build network architecture
        for hidden_dim in proj_dim:
            # BiMap layer
            layers.append(BiMapLayer(current_dim, hidden_dim))
            # ReEig layer for non-linearity
            layers.append(ReEigLayer())
            current_dim = hidden_dim

        # Log mapping before final layer
        layers.append(LogEigLayer())

        self.layers = nn.ModuleList(layers)

        # Final fully connected layer for classification
        # After LogEig, we vectorize the upper triangular part
        vector_dim = current_dim * (current_dim + 1) // 2
        self.fc = nn.Linear(vector_dim, num_classes)

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            X: Input SPD matrices (batch_size, n, n)
        Returns:
            logits: (batch_size, num_classes)
        """
        for layer in self.layers:
            X = layer(X)

        # Vectorize upper triangular part (including diagonal)
        # batch_size = X.size(0)
        n = X.size(1)

        # Extract upper triangular indices
        triu_indices = torch.triu_indices(n, n, offset=0)
        X_vec = X[:, triu_indices[0], triu_indices[1]]

        # Final classification
        logits = self.fc(X_vec)

        return logits


# Custom optimizer for Stiefel manifold
class StiefelSGD(torch.optim.Optimizer):
    """
    SGD optimizer with Stiefel manifold retraction
    """
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Check if parameter should be on Stiefel manifold
                # (typically BiMap weights)
                if hasattr(p, '_on_stiefel') and p._on_stiefel:
                    # Apply retraction
                    p.data = retraction_stiefel(p.data, p.grad.data, lr)
                else:
                    # Standard SGD update
                    p.data.add_(p.grad.data, alpha=-lr)

        return loss

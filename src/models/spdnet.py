from __future__ import annotations

import torch
import torch.nn as nn

from src.manifolds.spd_ops import re_eig, log_map_spd


# class BiReBlock(nn.Module): 
#     """
#     BiMap layer: 
#     X ∈ R^{B×d_in×d_in} → Y ∈ R^{B×d_out×d_out} 
#     Y_b = W X_b W^T
#     """ 
#     def __init__(
#         self,
#         d_in: int,
#         d_out: int,
#         eps: float = 1e-4,
#     ):
#         super().__init__()
#         self.d_in = d_in
#         self.d_out = d_out
#         self.eps = eps
#         self.W = nn.Parameter(torch.randn(d_out, d_in))
#         self.reset_parameters()

#     def reset_parameters(self):
#         # Inicialización semi-ortogonal (filas ortonormales)
#         with torch.no_grad():
#             # QR sobre Wᵀ: (d_in, d_out) -> q: (d_in, d_out)
#             q, _ = torch.linalg.qr(self.W.t())  # columnas ortonormales en q
#             self.W.copy_(q.t())                 # filas ortonormales en W

#     def forward(
#         self,
#         X: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         X: (B, d_in, d_in) return: (B, d_out, d_out)
#         """
#         # (paper huang andd gool)
#         # -------------------------------------------------
#         # 1. Capa BiMap: X -> W X W^T
#         # -------------------------------------------------
#         # Proyección a Stiefel en cada forward (aprox Riemanniano)
#         W = self.W
#         q, _ = torch.linalg.qr(W.t())    # (d_in, d_out)
#         W_st = q.t()                     # (d_out, d_in)

#         # WX: (B, d_in, d_out)
#         WX = torch.matmul(W_st, X)

#         # Y = WWXᵀ: (B, d_out, d_out)
#         Y = torch.matmul(WX, W.t())

#         if not torch.isfinite(Y).all():
#             print("NaN in BiMap")
#             raise SystemExit

#         # -------------------------------------------------
#         # 2. Capa ReEig (clamp de autovalores)
#         # -------------------------------------------------
#         return re_eig(Y, eps=self.eps)


# class SPDNet(nn.Module):
#     """
#     SPDNet con logaritmo matricial vía eigen-descomposición (PyTorch puro).

#     Arquitectura:
#       X SPD (d_in×d_in)
#         -> BiMap(d_in -> proj_dim)
#         -> ReEig
#         -> BiMap(d_in -> proj_dim)
#         -> ReEig
#         -> BiMap(d_in -> proj_dim)
#         -> ReEig
#         -> Log
#         -> Flatten
#         -> Linear -> num_classes
#     """

#     def __init__(
#         self,
#         d_in: int,
#         proj_dim: int = [70, 50, 30],
#         num_classes: int = 70,
#         metric: str = "log-euclidean",
#     ):
#         super().__init__()
#         self.d_in = d_in
#         self.proj_dim = proj_dim
#         self.num_classes = num_classes
#         self.metric = metric

#         # layers
#         self.birelayer1 = BiReBlock(d_in, proj_dim[0])
#         self.birelayer2 = BiReBlock(proj_dim[0], proj_dim[1])
#         self.birelayer3 = BiReBlock(proj_dim[1], proj_dim[2])
#         # Clasificador Euclídeo
#         d = proj_dim[-1]
#         self.fc = nn.Linear(d * d, num_classes)

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """
#         X: (B, d_in, d_in) SPD
#         """

#         # 3 bloques de BiRe (BiMap y Re)
#         X = self.birelayer1(X)
#         X = self.birelayer2(X)
#         X = self.birelayer3(X)

#         X = log_map_spd(X, metric=self.metric)

#         # Flatten + clasificación
#         B, d, _ = X.shape
#         X_flat = X.reshape(B, d * d)
#         out = self.fc(X_flat)
#         return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class BiMapFunction(Function):
    """BiMap layer: X_k = W_k^T X_{k-1} W_k"""
    
    @staticmethod
    def forward(ctx, X, W):
        """
        Args:
            X: Input SPD matrix (batch_size, n, n)
            W: Weight matrix on Stiefel manifold (n, m) where m <= n
        Returns:
            Output SPD matrix (batch_size, m, m)
        """
        ctx.save_for_backward(X, W)
        q, _ = torch.linalg.qr(W.t())    # (d_in, d_out)
        W_st = q.t()
        # X_k = W^T X_{k-1} W
        output = torch.matmul(torch.matmul(W_st, X), W_st.t())
        print("BiMap here", output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
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
        grad_X = torch.matmul(torch.matmul(W, grad_output), W.t())
        
        # Euclidean gradient w.r.t. W (Eq. 9)
        # ∇L^(k)_{W_k} = 2 (dL/dX_k) W_k X_{k-1}
        euclidean_grad = 2 * torch.matmul(X, torch.matmul(W, grad_output)) 
        
        # Convert to Riemannian gradient (Eq. 7)
        # ∇̃L^(k)_{W_k} = ∇L^(k)_{W_k} - ∇L^(k)_{W_k} (W_k)^T W_k
        print(euclidean_grad.shape, W.shape)
        grad_W = euclidean_grad - torch.matmul(
            torch.matmul(euclidean_grad, W.t()), W
        )
        
        return grad_X, grad_W


class ReEigFunction(Function):
    """ReEig layer: Rectifies eigenvalues to be >= epsilon"""
    
    @staticmethod
    def forward(ctx, X, epsilon=1e-4):
        """
        Args:
            X: Input SPD matrix (batch_size, n, n)
            epsilon: Threshold for rectification
        Returns:
            Output SPD matrix with rectified eigenvalues
        """
        # Para evitar errores
        epsilon = 1e-4
        X = X + epsilon * torch.eye(X.size(-1), device=X.device)
        
        if not torch.isfinite(X).all():
            print("NaN o Inf en X")
            print(X)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(X)
        
        # Rectify eigenvalues: max(epsilon, sigma_i)
        # eigenvalues_rectified = torch.maximum(
        #     torch.tensor(epsilon, device=X.device, dtype=X.dtype),
        #     eigenvalues
        # )
        eigenvalues_rectified = torch.clamp(eigenvalues, min=epsilon, max=1e6)
        
        ctx.save_for_backward(eigenvectors, eigenvalues, eigenvalues_rectified)
        ctx.epsilon = epsilon
        
        # Reconstruct: X_k = U max(I*epsilon, Sigma) U^T
        output = torch.matmul(
            torch.matmul(eigenvectors, torch.diag_embed(eigenvalues_rectified)),
            eigenvectors.transpose(-2, -1)
        )
        print("ReEig here", output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
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
    """LogEig layer: Takes logarithm of eigenvalues"""
    
    @staticmethod
    def forward(ctx, X):
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
        print("LogEig here", output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
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
        grad_X = 2 * torch.matmul(torch.matmul(U, hadamard_term), U.transpose(-2, -1))
        grad_X += torch.matmul(
            torch.matmul(U, torch.diag_embed(dL_dSigma_diag)),
            U.transpose(-2, -1)
        )
        
        return grad_X


def compute_P_matrix(eigenvalues, eps=1e-4):
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
    mask_diag = torch.eye(eigenvalues.size(-1), device=eigenvalues.device).bool()
    
    # Evitar división por cero
    diff_safe = diff + mask_diag.float() * eps
    
    # Suavizado con tanh para diferencias pequeñas
    smooth_factor = torch.tanh(torch.abs(diff_safe) / eps)
    P = smooth_factor / diff_safe
    
    # Set diagonal to zero
    P = P * (~mask_diag).float()
    
    # Clamp para evitar valores extremos
    P = torch.clamp(P, min=-1e4, max=1e4)
    
    return P


def retraction_stiefel(W, gradient, lr):
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
    Q, R = torch.linalg.qr(W_updated)
    
    return Q


class BiMapLayer(nn.Module):
    """BiMap Layer with Stiefel manifold constraint"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize on Stiefel manifold using QR decomposition
        W_init = torch.randn(output_dim, input_dim)
        W_init, _ = torch.linalg.qr(W_init.t())
        self.W = nn.Parameter(W_init)
        
    def forward(self, X):
        return BiMapFunction.apply(X, self.W)


class ReEigLayer(nn.Module):
    """ReEig Layer: Rectifies eigenvalues"""
    
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, X):
        return ReEigFunction.apply(X, self.epsilon)


class LogEigLayer(nn.Module):
    """LogEig Layer: Takes logarithm of eigenvalues"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return LogEigFunction.apply(X)


class SPDNet(nn.Module):
    """
    SPD Network for classification on SPD manifolds
    """

    def __init__(self, input_dim, hidden_dims, num_classes):
        """
        Args:
            input_dim: Dimension of input SPD matrices
            hidden_dims: List of hidden dimensions for BiMap layers
            num_classes: Number of output classes
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build network architecture
        for hidden_dim in hidden_dims:
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
        
    def forward(self, X):
        """
        Args:
            X: Input SPD matrices (batch_size, n, n)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Pass through SPD layers
        for layer in self.layers:
            X = layer(X)
        
        # Vectorize upper triangular part (including diagonal)
        batch_size = X.size(0)
        n = X.size(1)
        
        # Extract upper triangular indices
        triu_indices = torch.triu_indices(n, n, offset=0)
        X_vec = X[:, triu_indices[0], triu_indices[1]]
        
        # Final classification
        logits = self.fc(X_vec)
        
        return logits


# Custom optimizer for Stiefel manifold
class StiefelSGD(torch.optim.Optimizer):
    """SGD optimizer with Stiefel manifold retraction"""
    
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


# Example usage
# if __name__ == "__main__":
#     # Create sample data
#     batch_size = 8
#     input_dim = 10
#     num_classes = 5
    
#     # Random SPD matrices (ensure they are positive definite)
#     A = torch.randn(batch_size, input_dim, input_dim)
#     X = torch.matmul(A, A.transpose(-2, -1)) + torch.eye(input_dim) * 0.1
    
#     # Labels
#     y = torch.randint(0, num_classes, (batch_size,))
    
#     # Create network
#     model = SPDNet(
#         input_dim=input_dim,
#         hidden_dims=[8, 6],
#         num_classes=num_classes
#     )
    
#     # Mark BiMap weights as being on Stiefel manifold
#     for module in model.modules():
#         if isinstance(module, BiMapLayer):
#             module.W._on_stiefel = True
    
#     # Create optimizer
#     optimizer = StiefelSGD(model.parameters(), lr=0.01)
#     criterion = nn.CrossEntropyLoss()
    
#     # Training step
#     model.train()
#     optimizer.zero_grad()
    
#     output = model(X)
#     loss = criterion(output, y)
    
#     print(f"Output shape: {output.shape}")
#     print(f"Loss: {loss.item():.4f}")
    
#     loss.backward()
#     optimizer.step()
    
#     print("Backward pass completed successfully!")
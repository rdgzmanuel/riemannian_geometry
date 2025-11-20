"""
Grassmann Manifold Operations (Optimized & Vectorized)
"""

import torch


class GrassmannOps:
    @staticmethod
    def projection_metric(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        # X1, X2: (Batch, D, q)
        # Batch matrix multiplication
        proj1 = torch.bmm(X1, X1.transpose(-2, -1))
        proj2 = torch.bmm(X2, X2.transpose(-2, -1))
        diff = proj1 - proj2
        # Frobenius norm over last two dimensions
        return (0.5**0.5) * torch.norm(diff, p="fro", dim=(-2, -1))

    @staticmethod
    def qr_decomposition(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pytorch supports batched QR natively
        Q, R = torch.linalg.qr(X, mode="reduced")

        # Strict diagonal positivity enforcement (crucial for unique derivative)
        # Get diagonal elements
        diag_R = torch.diagonal(R, dim1=-2, dim2=-1)
        signs = torch.sign(diag_R)
        # Handle zeros by treating sign as 1
        signs[signs == 0] = 1

        # Expand for broadcasting: (Batch, 1, q) and (Batch, q, 1)
        Q = Q * signs.unsqueeze(-2)
        R = R * signs.unsqueeze(-1)
        return Q, R

    @staticmethod
    def qr_backward_reorth(
        grad_Q: torch.Tensor, Q: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized backward for ReOrth Layer where grad_R is known to be 0.
        Implements simplified Eq. 15 from the paper.
        """
        # Batched inverse
        R_inv = torch.linalg.inv(R)
        R_inv_T = R_inv.transpose(-2, -1)

        # S = I - Q @ Q^T
        batch_size, d, q = Q.shape
        I = (
            torch.eye(d, device=Q.device, dtype=Q.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        S = I - torch.bmm(Q, Q.transpose(-2, -1))

        # Term A: S^T @ grad_Q
        # Since S is symmetric, S^T = S
        A = torch.bmm(S, grad_Q)

        # Term B: Q @ (Q^T @ grad_Q)_{bsym}
        QT_gradQ = torch.bmm(Q.transpose(-2, -1), grad_Q)
        QT_gradQ_bsym = GrassmannOps._extract_below_symmetric(QT_gradQ)
        B = torch.bmm(Q, QT_gradQ_bsym)

        # Result = (A + B) @ R^{-T}
        grad_X = torch.bmm(A + B, R_inv_T)

        return grad_X

    @staticmethod
    def _extract_below_symmetric(A: torch.Tensor) -> torch.Tensor:
        # Batched extraction
        # A_bsym = A_tril - (A^T)_tril
        A_tril = torch.tril(A, diagonal=-1)
        AT_tril = torch.tril(A.transpose(-2, -1), diagonal=-1)
        return A_tril - AT_tril

    @staticmethod
    def eig_decomposition(
        X: torch.Tensor, top_k: int = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Batched eigh
        # X is (Batch, D, D) projection matrix (symmetric)
        # Ensure symmetry for numerical stability
        X_sym = 0.5 * (X + X.transpose(-2, -1))

        eigenvalues, eigenvectors = torch.linalg.eigh(X_sym)

        # Sort descending (linalg.eigh returns ascending)
        eigenvalues = torch.flip(eigenvalues, dims=[-1])
        eigenvectors = torch.flip(eigenvectors, dims=[-1])

        if top_k is not None:
            eigenvalues = eigenvalues[..., :top_k]
            eigenvectors = eigenvectors[..., :top_k]

        return eigenvectors, eigenvalues

    @staticmethod
    def eig_backward(
        grad_U: torch.Tensor, U: torch.Tensor, Sigma: torch.Tensor
    ) -> torch.Tensor:
        # Batched implementation of Eq 16
        batch_size, n, k = U.shape

        # Compute K matrix (Broadcasting)
        # Sigma: (Batch, k)
        s_i = Sigma.unsqueeze(-1)  # (Batch, k, 1)
        s_j = Sigma.unsqueeze(-2)  # (Batch, 1, k)
        denom = s_i - s_j

        # Avoid division by zero
        mask = torch.abs(denom) > 1e-8
        K = torch.zeros_like(denom)
        K[mask] = 1.0 / denom[mask]

        # We need to handle the case where k < n (subspace tracking)
        # The paper formula implies full expansion, but efficiently:

        # Term 1: U @ (K^T * (U^T @ grad_U)) @ U^T
        # This handles the rotation within the subspace
        UT_gradU = torch.bmm(U.transpose(-2, -1), grad_U)  # (Batch, k, k)
        inner = K.transpose(-2, -1) * UT_gradU
        term1 = torch.bmm(torch.bmm(U, inner), U.transpose(-2, -1))

        # Term 2: Off-diagonal/subspace projection part
        # Since we often only keep top_k, we need the projection to the orthogonal complement
        # Proj_perp = I - U U^T
        I = torch.eye(n, device=U.device).unsqueeze(0).expand(batch_size, -1, -1)
        UU_T = torch.bmm(U, U.transpose(-2, -1))
        Proj_perp = I - UU_T

        # The gradient flow to the orthogonal complement
        # Formula: (I - UU^T) @ grad_U @ diag(1/sigma) @ U^T
        # Note: This is an approximation if we don't have all eigenvectors.
        # Standard stable implementation for deep learning:
        term2 = torch.bmm(
            torch.bmm(Proj_perp, grad_U),
            (1.0 / Sigma.unsqueeze(-2)) * U.transpose(-2, -1),
        )

        # Combine
        return term1 + term2

    @staticmethod
    def projection_mapping(X: torch.Tensor) -> torch.Tensor:
        return torch.bmm(X, X.transpose(-2, -1))

    @staticmethod
    def retraction_psd(W: torch.Tensor) -> torch.Tensor:
        Q, _ = torch.linalg.qr(W, mode="reduced")
        # No sign correction needed for retraction usually, but consistency helps
        return Q

"""
Grassmann Manifold Operations
Implements manifold-specific operations for Grassmann manifolds.
"""

import torch


class GrassmannOps:
    """Operations on Grassmann manifolds Gr(q, D)."""

    @staticmethod
    def projection_metric(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Compute projection metric distance between two points on Grassmann manifold.

        Args:
            X1: Orthonormal matrix of shape (batch, D, q) or (D, q)
            X2: Orthonormal matrix of shape (batch, D, q) or (D, q)

        Returns:
            Distance scalar or batch of distances
        """
        proj1 = X1 @ X1.transpose(-2, -1)
        proj2 = X2 @ X2.transpose(-2, -1)
        diff = proj1 - proj2
        return (0.5**0.5) * torch.norm(diff, p="fro", dim=(-2, -1))

    @staticmethod
    def qr_decomposition(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        QR decomposition with orthonormalization.

        Args:
            X: Matrix of shape (..., m, n)

        Returns:
            Q: Orthonormal matrix (..., m, n)
            R: Upper triangular matrix (..., n, n)
        """
        Q, R = torch.linalg.qr(X, mode="reduced")
        # Ensure positive diagonal for consistency
        signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        signs[signs == 0] = 1
        Q = Q * signs.unsqueeze(-2)
        R = R * signs.unsqueeze(-1)
        return Q, R

    @staticmethod
    def qr_backward_q(
        grad_Q: torch.Tensor, Q: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """
        Backward pass through QR decomposition w.r.t. Q.
        From Equation 13 in the paper.

        Args:
            grad_Q: Gradient w.r.t. Q of shape (..., m, n)
            Q: Orthonormal matrix from forward pass (..., m, n)
            R: Upper triangular matrix from forward pass (..., n, n)

        Returns:
            Gradient w.r.t. input X
        """
        R_inv = torch.inverse(R)
        S = torch.eye(Q.shape[-2], device=Q.device, dtype=Q.dtype) - Q @ Q.transpose(
            -2, -1
        )

        # Compute Q^T @ grad_Q @ R^{-1}
        QT_gradQ_Rinv = Q.transpose(-2, -1) @ grad_Q @ R_inv

        # Extract antisymmetric part
        QT_gradQ_Rinv_asym = GrassmannOps._extract_antisymmetric(QT_gradQ_Rinv)

        grad_X = (
            S.transpose(-2, -1) @ grad_Q + Q @ QT_gradQ_Rinv_asym
        ) @ R_inv.transpose(-2, -1)
        return grad_X

    @staticmethod
    def qr_backward_r(
        grad_R: torch.Tensor, Q: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """
        Backward pass through QR decomposition w.r.t. R.
        From Equation 14 in the paper.

        Args:
            grad_R: Gradient w.r.t. R of shape (..., n, n)
            Q: Orthonormal matrix from forward pass (..., m, n)
            R: Upper triangular matrix from forward pass (..., n, n)

        Returns:
            Gradient w.r.t. input X
        """
        R_inv = torch.inverse(R)

        # Compute (grad_R @ R^T)
        gradR_RT = grad_R @ R.transpose(-2, -1)
        gradR_RT_bsym = GrassmannOps._extract_below_symmetric(gradR_RT)

        grad_X = Q @ (grad_R - gradR_RT_bsym @ R_inv.transpose(-2, -1))
        return grad_X

    @staticmethod
    def qr_backward_full(
        grad_Q: torch.Tensor, grad_R: torch.Tensor, Q: torch.Tensor, R: torch.Tensor
    ) -> torch.Tensor:
        """
        Full backward pass through QR decomposition.
        From Equation 15 in the paper.

        Args:
            grad_Q: Gradient w.r.t. Q
            grad_R: Gradient w.r.t. R
            Q: Orthonormal matrix from forward pass
            R: Upper triangular matrix from forward pass

        Returns:
            Gradient w.r.t. input X
        """
        R_inv = torch.inverse(R)
        S = torch.eye(Q.shape[-2], device=Q.device, dtype=Q.dtype) - Q @ Q.transpose(
            -2, -1
        )

        # First term from grad_Q
        QT_gradQ_Rinv = Q.transpose(-2, -1) @ grad_Q @ R_inv
        QT_gradQ_Rinv_bsym = GrassmannOps._extract_below_symmetric(QT_gradQ_Rinv)

        term1 = (
            S.transpose(-2, -1) @ grad_Q + Q @ QT_gradQ_Rinv_bsym
        ) @ R_inv.transpose(-2, -1)

        # Second term from grad_R
        gradR_RT = grad_R @ R.transpose(-2, -1)
        gradR_RT_bsym = GrassmannOps._extract_below_symmetric(gradR_RT)

        term2 = Q @ (grad_R - gradR_RT_bsym @ R_inv.transpose(-2, -1))

        return term1 + term2

    @staticmethod
    def _extract_antisymmetric(A: torch.Tensor) -> torch.Tensor:
        """
        Extract antisymmetric part: A_asym = A_tril - (A_tril)^T

        Args:
            A: Square matrix (..., n, n)

        Returns:
            Antisymmetric matrix
        """
        A_tril = torch.tril(A, diagonal=-1)
        return A_tril - A_tril.transpose(-2, -1)

    @staticmethod
    def _extract_below_symmetric(A: torch.Tensor) -> torch.Tensor:
        """
        Extract below-symmetric part: A_bsym = A_tril - (A^T)_tril

        Args:
            A: Square matrix (..., n, n)

        Returns:
            Below-symmetric matrix
        """
        A_tril = torch.tril(A, diagonal=-1)
        AT_tril = torch.tril(A.transpose(-2, -1), diagonal=-1)
        return A_tril - AT_tril

    @staticmethod
    def eig_decomposition(
        X: torch.Tensor, top_k: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Eigenvalue decomposition for symmetric matrices.

        Args:
            X: Symmetric matrix (..., n, n)
            top_k: Number of top eigenvectors to return (default: all)

        Returns:
            eigenvectors: (..., n, k) where k = top_k or n
            eigenvalues: (..., k)
        """
        # Ensure symmetry
        X_sym = 0.5 * (X + X.transpose(-2, -1))

        eigenvalues, eigenvectors = torch.linalg.eigh(X_sym)

        # Sort in descending order
        idx = torch.argsort(eigenvalues, dim=-1, descending=True)
        eigenvalues = torch.gather(eigenvalues, -1, idx)
        eigenvectors = torch.gather(
            eigenvectors, -1, idx.unsqueeze(-2).expand_as(eigenvectors)
        )

        if top_k is not None:
            eigenvalues = eigenvalues[..., :top_k]
            eigenvectors = eigenvectors[..., :top_k]

        return eigenvectors, eigenvalues

    @staticmethod
    def eig_backward(
        grad_U: torch.Tensor,
        grad_Sigma: torch.Tensor,
        U: torch.Tensor,
        Sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward pass through eigenvalue decomposition.
        From Proposition 1 in the paper (Equation 16).

        Args:
            grad_U: Gradient w.r.t. eigenvectors (..., n, k)
            grad_Sigma: Gradient w.r.t. eigenvalues (..., k)
            U: Eigenvectors from forward pass (..., n, k)
            Sigma: Eigenvalues from forward pass (..., k)

        Returns:
            Gradient w.r.t. input X
        """
        n = U.shape[-2]
        k = U.shape[-1]

        # Pad U and grad_U if k < n
        if k < n:
            U_full = torch.cat(
                [U, torch.zeros(*U.shape[:-1], n - k, device=U.device, dtype=U.dtype)],
                dim=-1,
            )
            grad_U_full = torch.cat(
                [
                    grad_U,
                    torch.zeros(
                        *grad_U.shape[:-1],
                        n - k,
                        device=grad_U.device,
                        dtype=grad_U.dtype,
                    ),
                ],
                dim=-1,
            )
        else:
            U_full = U
            grad_U_full = grad_U

        # Compute K matrix: K_ij = 1/(sigma_i - sigma_j) for i != j, else 0
        sigma_i = Sigma.unsqueeze(-1)  # (..., k, 1)
        sigma_j = Sigma.unsqueeze(-2)  # (..., 1, k)

        K = torch.zeros(
            *Sigma.shape, Sigma.shape[-1], device=Sigma.device, dtype=Sigma.dtype
        )
        mask = torch.abs(sigma_i - sigma_j) > 1e-8
        K = torch.where(mask, 1.0 / (sigma_i - sigma_j), K)

        # Pad K and grad_Sigma
        if k < n:
            K_full = torch.zeros(*K.shape[:-2], n, n, device=K.device, dtype=K.dtype)
            K_full[..., :k, :k] = K

            grad_Sigma_full = torch.zeros(
                *grad_Sigma.shape[:-1],
                n,
                device=grad_Sigma.device,
                dtype=grad_Sigma.dtype,
            )
            grad_Sigma_full[..., :k] = grad_Sigma
        else:
            K_full = K
            grad_Sigma_full = grad_Sigma

        # Compute gradient U @ (K^T ∘ (U^T @ grad_U)) @ U^T + U @ diag(grad_Sigma) @ U^T
        UT_gradU = U_full.transpose(-2, -1) @ grad_U_full
        term1 = (
            U_full @ (K_full.transpose(-2, -1) * UT_gradU) @ U_full.transpose(-2, -1)
        )
        term2 = U_full @ torch.diag_embed(grad_Sigma_full) @ U_full.transpose(-2, -1)

        grad_X = term1 + term2
        return grad_X

    @staticmethod
    def projection_mapping(X: torch.Tensor) -> torch.Tensor:
        """
        Project orthonormal matrix to projection matrix: X @ X^T

        Args:
            X: Orthonormal matrix (..., D, q)

        Returns:
            Projection matrix (..., D, D)
        """
        return X @ X.transpose(-2, -1)

    @staticmethod
    def retraction_psd(W: torch.Tensor) -> torch.Tensor:
        """
        Retraction operation on PSD manifold.
        Maps from tangent space back to manifold.

        Args:
            W: Matrix in tangent space (..., m, n)

        Returns:
            Retracted matrix on PSD manifold
        """
        # Use QR retraction for numerical stability
        Q, _ = GrassmannOps.qr_decomposition(W)
        return Q

    @staticmethod
    def geodesic_distance(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic distance on Grassmann manifold using projection metric.

        Args:
            X1: Orthonormal matrix (..., D, q)
            X2: Orthonormal matrix (..., D, q)

        Returns:
            Distance
        """
        return GrassmannOps.projection_metric(X1, X2)

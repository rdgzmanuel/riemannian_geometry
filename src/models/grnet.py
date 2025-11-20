"""
Grassmann Network Layers
Custom PyTorch layers with manual forward/backward passes for Grassmann manifolds.
"""

import torch
import torch.nn as nn
from src.manifolds.grassmann_ops import GrassmannOps


class FRMapFunction(torch.autograd.Function):
    """Full Rank Mapping Layer with custom backward."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y = W @ X

        Args:
            X: Input orthonormal matrix (batch, d_in, q)
            W: Weight matrix (d_out, d_in) - full rank

        Returns:
            Y: Output matrix (batch, d_out, q)
        """
        Y = torch.bmm(W.unsqueeze(0).expand(X.shape[0], -1, -1), X)
        ctx.save_for_backward(X, W)
        return Y

    @staticmethod
    def backward(ctx, grad_Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass for FRMap layer.

        Args:
            grad_Y: Gradient w.r.t. output (batch, d_out, q)

        Returns:
            grad_X: Gradient w.r.t. input
            grad_W: Gradient w.r.t. weights (for PSD manifold optimization)
        """
        X, W = ctx.saved_tensors

        # Gradient w.r.t. X: W^T @ grad_Y
        grad_X = torch.bmm(
            W.unsqueeze(0).expand(grad_Y.shape[0], -1, -1).transpose(-2, -1), grad_Y
        )

        # Gradient w.r.t. W: sum over batch of grad_Y @ X^T
        # This is the Euclidean gradient, will be projected to PSD manifold in optimizer
        grad_W = torch.sum(torch.bmm(grad_Y, X.transpose(-2, -1)), dim=0)

        return grad_X, grad_W


class ReOrthFunction(torch.autograd.Function):
    """Re-Orthonormalization Layer using QR decomposition."""

    @staticmethod
    def forward(ctx, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Q, R = QR(X), return Q

        Args:
            X: Input matrix (batch, d, q)

        Returns:
            Q: Orthonormal matrix (batch, d, q)
        """
        batch_size = X.shape[0]
        Q_list, R_list = [], []

        for i in range(batch_size):
            Q, R = GrassmannOps.qr_decomposition(X[i])
            Q_list.append(Q)
            R_list.append(R)

        Q = torch.stack(Q_list, dim=0)
        R = torch.stack(R_list, dim=0)

        ctx.save_for_backward(Q, R)
        return Q

    @staticmethod
    def backward(ctx, grad_Q: torch.Tensor) -> torch.Tensor:
        """
        Backward pass through QR decomposition.
        Uses Equation 15 from the paper.

        Args:
            grad_Q: Gradient w.r.t. Q (batch, d, q)

        Returns:
            grad_X: Gradient w.r.t. input X
        """
        Q, R = ctx.saved_tensors
        batch_size = grad_Q.shape[0]

        grad_X_list = []
        for i in range(batch_size):
            # Since we only care about Q in the forward pass, grad_R = 0
            grad_R = torch.zeros_like(R[i])
            grad_X = GrassmannOps.qr_backward_full(grad_Q[i], grad_R, Q[i], R[i])
            grad_X_list.append(grad_X)

        grad_X = torch.stack(grad_X_list, dim=0)
        return grad_X


class ProjMapFunction(torch.autograd.Function):
    """Projection Mapping Layer: X @ X^T"""

    @staticmethod
    def forward(ctx, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: P = X @ X^T

        Args:
            X: Orthonormal matrix (batch, d, q)

        Returns:
            P: Projection matrix (batch, d, d)
        """
        P = GrassmannOps.projection_mapping(X)
        ctx.save_for_backward(X)
        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor) -> torch.Tensor:
        """
        Backward pass through projection mapping.

        d(X @ X^T) = dX @ X^T + X @ dX^T
        => grad_X = grad_P @ X + grad_P^T @ X = 2 * grad_P @ X (since grad_P is symmetric)

        Args:
            grad_P: Gradient w.r.t. P (batch, d, d)

        Returns:
            grad_X: Gradient w.r.t. X
        """
        (X,) = ctx.saved_tensors

        # Make grad_P symmetric
        grad_P_sym = 0.5 * (grad_P + grad_P.transpose(-2, -1))

        # grad_X = 2 * grad_P @ X
        grad_X = 2.0 * torch.bmm(grad_P_sym, X)

        return grad_X


class ProjPoolingFunction(torch.autograd.Function):
    """Projection Pooling: Mean pooling on projection matrices."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, n: int) -> torch.Tensor:
        """
        Forward pass: Average n projection matrices or spatial pooling.

        Args:
            X: Input projection matrices (batch, n, d, d) or (batch, d, d)
            n: Number of instances to pool

        Returns:
            Y: Pooled projection matrix (batch, d, d)
        """
        if X.dim() == 4:  # Across projection matrices
            Y = torch.mean(X, dim=1)
            ctx.n_type = "across"
        else:  # Within projection matrix (spatial pooling)
            # Reshape for spatial pooling
            batch, d, _ = X.shape
            patch_size = int(n**0.5)

            # Use adaptive avg pooling
            Y = torch.nn.functional.adaptive_avg_pool2d(
                X, (d // patch_size, d // patch_size)
            )
            ctx.n_type = "spatial"
            ctx.original_size = d
            ctx.patch_size = patch_size

        ctx.n = n
        return Y

    @staticmethod
    def backward(ctx, grad_Y: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass through pooling.

        Args:
            grad_Y: Gradient w.r.t. output

        Returns:
            grad_X: Gradient w.r.t. input
            None: for n parameter
        """
        if ctx.n_type == "across":
            # Gradient is broadcast to all n inputs
            grad_X = grad_Y.unsqueeze(1).expand(-1, ctx.n, -1, -1) / ctx.n
        else:
            # Spatial unpooling
            batch = grad_Y.shape[0]
            grad_X = torch.nn.functional.interpolate(
                grad_Y,
                size=(ctx.original_size, ctx.original_size),
                mode="bilinear",
                align_corners=False,
            )
            grad_X = grad_X / ctx.n

        return grad_X, None


class OrthMapFunction(torch.autograd.Function):
    """Orthonormal Mapping: Extract top eigenvectors from projection matrix."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, q: int) -> torch.Tensor:
        """
        Forward pass: Compute top q eigenvectors of X.

        Args:
            X: Projection matrix (batch, d, d)
            q: Number of eigenvectors to extract

        Returns:
            U: Top q eigenvectors (batch, d, q)
        """
        batch_size = X.shape[0]
        U_list, Sigma_list = [], []

        for i in range(batch_size):
            U, Sigma = GrassmannOps.eig_decomposition(X[i], top_k=q)
            U_list.append(U)
            Sigma_list.append(Sigma)

        U = torch.stack(U_list, dim=0)
        Sigma = torch.stack(Sigma_list, dim=0)

        ctx.save_for_backward(U, Sigma)
        ctx.q = q
        return U

    @staticmethod
    def backward(ctx, grad_U: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass through eigenvalue decomposition.
        Uses Proposition 1 from the paper (Equation 16).

        Args:
            grad_U: Gradient w.r.t. U (batch, d, q)

        Returns:
            grad_X: Gradient w.r.t. input X
            None: for q parameter
        """
        U, Sigma = ctx.saved_tensors
        batch_size = grad_U.shape[0]

        grad_X_list = []
        for i in range(batch_size):
            grad_Sigma = torch.zeros_like(Sigma[i])  # We only use U, not Sigma
            grad_X = GrassmannOps.eig_backward(grad_U[i], grad_Sigma, U[i], Sigma[i])
            grad_X_list.append(grad_X)

        grad_X = torch.stack(grad_X_list, dim=0)
        return grad_X, None


# ==================== Layer Modules ====================


class FRMapLayer(nn.Module):
    """Full Rank Mapping Layer."""

    def __init__(self, d_in: int, d_out: int, num_maps: int = 1):
        """
        Initialize FRMap layer.

        Args:
            d_in: Input dimension
            d_out: Output dimension (must be < d_in)
            num_maps: Number of parallel transformations
        """
        super().__init__()
        assert d_out < d_in, "d_out must be less than d_in for dimensionality reduction"

        self.d_in = d_in
        self.d_out = d_out
        self.num_maps = num_maps

        # Initialize weights as random full rank matrices
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(d_out, d_in) / (d_in**0.5))
            for _ in range(num_maps)
        ])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Input (batch, d_in, q) or list of (batch, d_in, q) if num_maps > 1

        Returns:
            Y: Output (batch, d_out, q) or list for multiple maps
        """
        if self.num_maps == 1:
            return FRMapFunction.apply(X, self.weights[0])
        else:
            # Apply each weight matrix
            outputs = [FRMapFunction.apply(X, W) for W in self.weights]
            return outputs


class ReOrthLayer(nn.Module):
    """Re-Orthonormalization Layer using QR decomposition."""

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Input matrix (batch, d, q) or list of matrices

        Returns:
            Q: Orthonormal matrix (batch, d, q) or list
        """
        if isinstance(X, list):
            return [ReOrthFunction.apply(x) for x in X]
        else:
            return ReOrthFunction.apply(X)


class ProjMapLayer(nn.Module):
    """Projection Mapping Layer."""

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Orthonormal matrix (batch, d, q) or list

        Returns:
            P: Projection matrix (batch, d, d) or list
        """
        if isinstance(X, list):
            return [ProjMapFunction.apply(x) for x in X]
        else:
            return ProjMapFunction.apply(X)


class ProjPoolingLayer(nn.Module):
    """Projection Pooling Layer."""

    def __init__(self, n: int = 4, pooling_type: str = "across"):
        """
        Initialize projection pooling.

        Args:
            n: Number of instances to pool
            pooling_type: 'across' for across matrices, 'spatial' for within matrix
        """
        super().__init__()
        self.n = n
        self.pooling_type = pooling_type

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Input projection matrices

        Returns:
            Y: Pooled projection matrix
        """
        return ProjPoolingFunction.apply(X, self.n)


class OrthMapLayer(nn.Module):
    """Orthonormal Mapping Layer."""

    def __init__(self, q: int):
        """
        Initialize orthonormal mapping.

        Args:
            q: Number of eigenvectors to extract
        """
        super().__init__()
        self.q = q

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X: Projection matrix (batch, d, d) or list

        Returns:
            U: Top q eigenvectors (batch, d, q) or list
        """
        if isinstance(X, list):
            return [OrthMapFunction.apply(x, self.q) for x in X]
        else:
            return OrthMapFunction.apply(X, self.q)


# ==================== Block Modules ====================


class ProjectionBlock(nn.Module):
    """Projection Block: FRMap + ReOrth layers."""

    def __init__(self, d_in: int, d_out: int, num_maps: int = 16):
        super().__init__()
        self.frmap = FRMapLayer(d_in, d_out, num_maps)
        self.reorth = ReOrthLayer()
        self.num_maps = num_maps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward through projection block."""
        Y = self.frmap(X)
        Y = self.reorth(Y)

        # If multiple maps, concatenate along batch dimension
        if isinstance(Y, list):
            Y = torch.cat(Y, dim=0)

        return Y


class PoolingBlock(nn.Module):
    """Pooling Block: ProjMap + ProjPooling + OrthMap."""

    def __init__(self, q: int, n: int = 4, pooling_type: str = "across"):
        super().__init__()
        self.projmap = ProjMapLayer()
        self.projpool = ProjPoolingLayer(n, pooling_type)
        self.orthmap = OrthMapLayer(q)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward through pooling block."""
        P = self.projmap(X)
        P_pooled = self.projpool(P)
        Y = self.orthmap(P_pooled)
        return Y


class OutputBlock(nn.Module):
    """Output Block: ProjMap + FC + Softmax."""

    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.projmap = ProjMapLayer()
        self.fc = nn.Linear(d * d, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward through output block."""
        P = self.projmap(X)

        # Flatten projection matrices
        batch_size = P.shape[0]
        P_flat = P.reshape(batch_size, -1)

        # FC layer
        logits = self.fc(P_flat)

        return logits


class GrNet(nn.Module):
    """
    Grassmann Network for learning on Grassmann manifolds.

    Architecture:
        - Multiple Projection Blocks (FRMap + ReOrth)
        - Optional Pooling Blocks (ProjMap + ProjPool + OrthMap)
        - Output Block (ProjMap + FC + Softmax)
    """

    def __init__(
        self,
        input_dim: int,
        q: int,
        num_classes: int,
        hidden_dims: list[int],
        num_maps: int = 16,
        use_pooling: bool = True,
        pooling_n: int = 4,
        pooling_type: str = "across",
    ):
        """
        Initialize GrNet.

        Args:
            input_dim: Dimension of input Grassmann point (D in Gr(q, D))
            q: Order of subspace
            num_classes: Number of output classes
            hidden_dims: List of hidden dimensions for each projection block
            num_maps: Number of parallel transformations per FRMap layer
            use_pooling: Whether to use pooling blocks
            pooling_n: Number of instances for pooling
            pooling_type: 'across' or 'spatial'
        """
        super().__init__()

        self.input_dim = input_dim
        self.q = q
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.num_maps = num_maps
        self.use_pooling = use_pooling

        # Build projection blocks
        self.projection_blocks = nn.ModuleList()
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            self.projection_blocks.append(
                ProjectionBlock(current_dim, hidden_dim, num_maps)
            )
            current_dim = hidden_dim

        # Optional pooling block
        if use_pooling:
            self.pooling_block = PoolingBlock(q, pooling_n, pooling_type)
        else:
            self.pooling_block = None

        # Output block
        self.output_block = OutputBlock(current_dim, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GrNet.

        Args:
            X: Input Grassmann points (batch, input_dim, q)

        Returns:
            logits: Class logits (batch, num_classes)
        """
        # Through projection blocks
        for block in self.projection_blocks:
            X = block(X)

        # Optional pooling
        if self.pooling_block is not None:
            X = self.pooling_block(X)

        # Output
        logits = self.output_block(X)

        return logits

    def get_manifold_parameters(self) -> list[nn.Parameter]:
        """
        Get parameters that lie on manifolds (for natural gradient optimizer).

        Returns:
            List of manifold parameters
        """
        manifold_params = []

        for block in self.projection_blocks:
            manifold_params.extend(block.frmap.weights)

        return manifold_params

    def get_euclidean_parameters(self) -> list[nn.Parameter]:
        """
        Get Euclidean parameters (FC layer weights).

        Returns:
            List of Euclidean parameters
        """
        return list(self.output_block.fc.parameters())


class GrNet1Block(GrNet):
    """GrNet with 1 projection block."""

    def __init__(
        self,
        input_dim: int,
        q: int,
        num_classes: int,
        hidden_dim: int = 100,
        num_maps: int = 16,
        use_pooling: bool = False,
    ):
        super().__init__(
            input_dim=input_dim,
            q=q,
            num_classes=num_classes,
            hidden_dims=[hidden_dim],
            num_maps=num_maps,
            use_pooling=use_pooling,
        )


class GrNet2Blocks(GrNet):
    """GrNet with 2 projection blocks."""

    def __init__(
        self,
        input_dim: int,
        q: int,
        num_classes: int,
        hidden_dim1: int = 300,
        hidden_dim2: int = 100,
        num_maps: int = 16,
        use_pooling: bool = False,
    ):
        super().__init__(
            input_dim=input_dim,
            q=q,
            num_classes=num_classes,
            hidden_dims=[hidden_dim1, hidden_dim2],
            num_maps=num_maps,
            use_pooling=use_pooling,
        )


def create_grnet(num_blocks: int = 2, num_classes: int = 130) -> GrNet:
    """
    Create GrNet configured for HDM05 dataset.

    Args:
        num_blocks: Number of projection blocks (1 or 2)
        num_classes: Number of action classes

    Returns:
        Configured GrNet model
    """
    config = {
        "input_dim": 93,  # Covariance descriptor size
        "q": 10,  # Subspace order
        "num_classes": num_classes,
        "hidden_dims": [80, 30],  # From paper: 93x80, 40x30
        "num_maps": 16,
        "use_pooling": True,
        "pooling_n": 4,
    }

    if num_blocks == 1:
        return GrNet1Block(
            input_dim=config["input_dim"],
            q=config["q"],
            num_classes=config["num_classes"],
            hidden_dim=60,  # From paper
            num_maps=config["num_maps"],
            use_pooling=False,
        )
    elif num_blocks == 2:
        return GrNet2Blocks(
            input_dim=config["input_dim"],
            q=config["q"],
            num_classes=config["num_classes"],
            hidden_dim1=80,  # From paper
            hidden_dim2=30,  # From paper
            num_maps=config["num_maps"],
            use_pooling=False,
        )
    else:
        return GrNet(
            input_dim=config["input_dim"],
            q=config["q"],
            num_classes=config["num_classes"],
            hidden_dims=config["hidden_dims"],
            num_maps=config["num_maps"],
            use_pooling=config["use_pooling"],
        )


def initialize_grnet_weights(model: GrNet) -> None:
    """
    Initialize GrNet weights as in the paper.

    Args:
        model: GrNet model to initialize
    """
    # FRMap weights are already initialized as random full rank matrices
    # in the FRMapLayer constructor

    # Initialize FC layer weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

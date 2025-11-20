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
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass for FRMap layer.

        Args:
            grad_output: Gradient w.r.t. output (batch, d_out, q)

        Returns:
            grad_X: Gradient w.r.t. input
            grad_W: Gradient w.r.t. weights (for PSD manifold optimization)
        """
        X, W = ctx.saved_tensors

        # Gradient w.r.t. X: W^T @ grad_output
        grad_X = torch.bmm(
            W.unsqueeze(0).expand(grad_output.shape[0], -1, -1).transpose(-2, -1),
            grad_output,
        )

        # Gradient w.r.t. W: sum over batch of grad_output @ X^T
        # This is the Euclidean gradient, will be projected to PSD manifold in optimizer
        grad_W = torch.sum(torch.bmm(grad_output, X.transpose(-2, -1)), dim=0)

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
        Q, R = GrassmannOps.qr_decomposition(X)
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
        grad_X = GrassmannOps.qr_backward_reorth(grad_Q, Q, R)
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
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """
        Backward pass through pooling.

        Args:
            grad_output: Gradient w.r.t. output

        Returns:
            grad_X: Gradient w.r.t. input
            None: for n parameter
        """
        if ctx.n_type == "across":
            # Gradient is broadcast to all n inputs
            grad_X = grad_output.unsqueeze(1).expand(-1, ctx.n, -1, -1) / ctx.n
        else:
            # Spatial unpooling
            batch = grad_output.shape[0]
            grad_X = torch.nn.functional.interpolate(
                grad_output,
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
        U, Sigma = GrassmannOps.eig_decomposition(X, top_k=q)
        ctx.save_for_backward(U, Sigma)
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
        grad_X = GrassmannOps.eig_backward(grad_U, U, Sigma)
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


class GrNetBlock(nn.Module):
    """
    A unified block that performs:
    1. Projection (FRMap -> ReOrth)
    2. Optional Pooling (ProjMap -> Pooling -> OrthMap)
    """

    def __init__(self, d_in, d_out, q, num_maps=16, use_pooling=False, pooling_n=4):
        super().__init__()
        self.projection = ProjectionBlock(d_in, d_out, num_maps)
        self.use_pooling = use_pooling

        if use_pooling:
            self.pooling_block = PoolingBlock(q, pooling_n, pooling_type="spatial")

    def forward(self, X):
        # X: (Batch, d_in, q)
        X = self.projection(X)  # -> (Batch, d_out, q)

        if self.use_pooling:
            X = self.pooling_block(X)
            # Output dimension will change based on pooling_n
            # If spatial pooling with n=4, d_out becomes d_out/2

        return X


class GrNet(nn.Module):
    def __init__(self, input_dim, q, num_classes, config):
        super().__init__()
        self.blocks = nn.ModuleList()

        current_dim = input_dim

        # config is a list of dicts describing the blocks
        for block_cfg in config:
            d_out = block_cfg["d_out"]
            use_pool = block_cfg.get("use_pooling", False)
            pool_n = block_cfg.get("pool_n", 4)

            block = GrNetBlock(
                d_in=current_dim,
                d_out=d_out,
                q=q,
                use_pooling=use_pool,
                pooling_n=pool_n,
            )
            self.blocks.append(block)

            # Update dimensions for next layer
            current_dim = d_out
            if use_pool:
                # Assuming spatial pooling W-ProjPooling
                patch_size = int(pool_n**0.5)
                current_dim = current_dim // patch_size

        self.output_block = OutputBlock(current_dim, num_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GrNet.

        Args:
            X: Input Grassmann points (batch, input_dim, q)

        Returns:
            logits: Class logits (batch, num_classes)
        """
        for block in self.blocks:
            X = block(X)
        return self.output_block(X)

    def get_manifold_parameters(self) -> list[nn.Parameter]:
        """
        Get parameters that lie on manifolds (for natural gradient optimizer).

        Returns:
            List of manifold parameters
        """
        manifold_params = []

        for block in self.blocks:
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


def create_grnet(num_blocks=2):
    """
    Strict replication of HDM05 architecture from paper.
    """
    q: int = 10
    input_dim: int = 93

    if num_blocks == 2:
        # Paper: GrNet-2Blocks: 93x80, 40x30
        # This implies:
        # 1. 93 -> 80
        # 2. Pooling (n=4, patch 2x2) reduces 80 -> 40
        # 3. 40 -> 30
        config = [
            {"d_out": 80, "use_pooling": True, "pool_n": 4},
            {"d_out": 30, "use_pooling": False},
        ]
    else:
        # GrNet-1Block: 93x60
        config = [{"d_out": 60, "use_pooling": False}]

    return GrNet(input_dim, q, num_classes=130, config=config)


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

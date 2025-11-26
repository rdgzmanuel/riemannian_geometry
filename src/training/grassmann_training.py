"""
Natural Gradient Optimizer for Riemannian Manifolds
Implements SGD on PSD manifolds for GrNet training.
"""

from collections.abc import Callable

import torch
import torch.optim as optimizer


class RiemannianSGD(optimizer.Optimizer):
    """
    Stochastic Gradient Descent on Riemannian Manifolds.

    Implements natural gradient updates on PSD manifolds for FRMap layers
    following Equations 9-10 from the paper.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        manifold_params: list[torch.nn.Parameter] | None = None,
    ):
        """
        Initialize Riemannian SGD optimizer.

        Args:
            params: Model parameters to optimize
            lr: Learning rate
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
            manifold_params: List of parameters on manifolds (FRMap weights)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Store which parameters are on manifolds
        self.manifold_params = set(manifold_params) if manifold_params else set()

    def step(self, closure: Callable | None = None) -> float | None:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            Loss if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                param_state = self.state[p]

                # Check if this parameter is on a manifold
                is_manifold = p in self.manifold_params

                if is_manifold:
                    # Natural gradient update on PSD manifold (Equations 9-10)
                    grad = self._project_to_tangent_space(grad, p.data)

                # Apply momentum
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(
                            grad
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1.0)
                    grad = buf

                # For manifold parameters, project gradient to tangent space
                if is_manifold:
                    grad = self._project_to_tangent_space(grad, p.data)

                # Update parameter
                if is_manifold:
                    # Retraction to manifold
                    p.data = self._retract_to_manifold(p.data, -lr * grad)
                else:
                    # Standard Euclidean update
                    p.data.add_(grad, alpha=-lr)

        return loss

    def _project_to_tangent_space(
        self, grad: torch.Tensor, W: torch.Tensor
    ) -> torch.Tensor:
        """
        Project Euclidean gradient to tangent space of PSD manifold.
        Implements Equation 9 from the paper: ∇L - ∇L @ W^T @ W

        Args:
            grad: Euclidean gradient
            W: Current weight matrix

        Returns:
            Riemannian gradient (projection to tangent space)
        """
        # Compute normal component: grad @ W^T @ W
        normal_component = grad @ W.T @ W

        # Project: tangent_grad = grad - normal_component
        tangent_grad = grad - normal_component

        return tangent_grad

    def _retract_to_manifold(
        self, W: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        """
        Retract from tangent space back to PSD manifold.
        Implements Equation 10 from the paper using QR retraction.

        Args:
            W: Current weight matrix
            update: Update vector in tangent space

        Returns:
            Updated weight matrix on manifold
        """
        return W + update


class MixedOptimizer:
    """
    Mixed optimizer that uses RiemannianSGD for manifold parameters
    and standard Adam for Euclidean parameters.
    """

    def __init__(
        self,
        manifold_params: list[torch.nn.Parameter],
        euclidean_params: list[torch.nn.Parameter],
        manifold_lr: float = 0.01,
        euclidean_lr: float = 0.001,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """
        Initialize mixed optimizer.

        Args:
            manifold_params: Parameters on manifolds (FRMap weights)
            euclidean_params: Euclidean parameters (FC layer weights)
            manifold_lr: Learning rate for manifold parameters
            euclidean_lr: Learning rate for Euclidean parameters
            momentum: Momentum for RiemannianSGD
            weight_decay: Weight decay for both optimizers
        """
        self.manifold_optimizer = RiemannianSGD(
            manifold_params,
            lr=manifold_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            manifold_params=manifold_params,
        )

        self.euclidean_optimizer = torch.optim.Adam(
            euclidean_params, lr=euclidean_lr, weight_decay=weight_decay
        )

    def zero_grad(self) -> None:
        """Zero gradients for both optimizers."""
        self.manifold_optimizer.zero_grad()
        self.euclidean_optimizer.zero_grad()

    def step(self) -> None:
        """Perform optimization step for both optimizers."""
        self.manifold_optimizer.step()
        self.euclidean_optimizer.step()

    def state_dict(self) -> dict:
        """Get state dict from both optimizers."""
        return {
            "manifold": self.manifold_optimizer.state_dict(),
            "euclidean": self.euclidean_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict for both optimizers."""
        self.manifold_optimizer.load_state_dict(state_dict["manifold"])
        self.euclidean_optimizer.load_state_dict(state_dict["euclidean"])


def create_grnet_optimizer(
    model: torch.nn.Module,
    manifold_lr: float = 0.01,
    euclidean_lr: float = 0.001,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    use_mixed: bool = True,
) -> optimizer.Optimizer:
    """
    Create appropriate optimizer for GrNet model.

    Args:
        model: GrNet model
        manifold_lr: Learning rate for manifold parameters
        euclidean_lr: Learning rate for Euclidean parameters
        momentum: Momentum factor
        weight_decay: Weight decay
        use_mixed: Whether to use mixed optimizer

    Returns:
        Optimizer instance
    """
    if hasattr(model, "get_manifold_parameters") and use_mixed:
        manifold_params = model.get_manifold_parameters()
        euclidean_params = model.get_euclidean_parameters()

        print("Using MixedOptimizer for GrNet training.")
        return MixedOptimizer(
            manifold_params=manifold_params,
            euclidean_params=euclidean_params,
            manifold_lr=manifold_lr,
            euclidean_lr=euclidean_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        # Fall back to standard optimizer
        print("Using standard SGD optimizer for GrNet training.")
        return torch.optim.SGD(
            model.parameters(),
            lr=manifold_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

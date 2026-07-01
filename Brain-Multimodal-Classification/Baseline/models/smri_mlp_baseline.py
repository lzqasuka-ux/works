"""
sMRI MLP Baseline Model
Simple MLP classifier for 3D sMRI volumes: AvgPool3d downsample → Flatten → MLP → binary classification.

This is the simplest baseline for sMRI brain classification — no convolution, no attention,
just global average pooling followed by fully-connected layers. Contrasts with the more
sophisticated sMRI-3D-SwinTransformer to measure the value of spatial structure modeling.

Input:  3D sMRI volume of shape (B, 1, D, H, W)
Output: Binary classification logits of shape (B, num_classes)
"""

import torch
import torch.nn as nn
from typing import Optional, List


class sMRIMlpClassifier(nn.Module):
    """
    Simple MLP baseline for sMRI brain classification.

    Architecture:
        3D Volume (B,1,D,H,W) → AvgPool3d → Flatten → MLP → logits (B, num_classes)

    The AvgPool3d aggressively downsamples the 3D volume (e.g., 96³ → 8³ = 512 values),
    discarding spatial structure, then MLP layers perform classification.
    This serves as a lower-bound baseline for the sMRI-3D-SwinTransformer.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int] = (96, 96, 96),
        pool_kernel: Optional[Tuple[int, int, int]] = None,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
        activation: nn.Module = nn.ReLU,
        use_batch_norm: bool = True,
        **kwargs
    ):
        """
        Args:
            volume_shape: Input 3D volume shape (D, H, W).
            pool_kernel: Kernel size for AvgPool3d. If None, uses (volume/8) to get ~512 features.
            hidden_dims: List of hidden layer dimensions. If None, uses [256, 128].
            num_classes: Number of output classes (2 for binary).
            dropout: Dropout rate after each hidden layer.
            activation: Activation function class.
            use_batch_norm: Whether to apply BatchNorm1d after each hidden Linear.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.volume_shape = volume_shape
        self.num_classes = num_classes

        # --- Adaptive Average Pooling to compress 3D volume ---
        if pool_kernel is None:
            # Target ~512 features: 8³ = 512
            pool_kernel = tuple(max(1, d // 8) for d in volume_shape)

        self.avgpool = nn.AvgPool3d(kernel_size=pool_kernel)

        # Compute flattened dimension after pooling
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *volume_shape)
            pooled = self.avgpool(dummy)
            input_dim = pooled.numel()

        # --- MLP layers ---
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Classification head
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) — 3D sMRI volume.

        Returns:
            logits: (B, num_classes) — raw classification scores.
        """
        x = self.avgpool(x)             # (B, C, D', H', W')
        x = x.view(x.shape[0], -1)      # (B, pooled_features)
        x = self.mlp(x)                  # (B, num_classes)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the penultimate layer (before final Linear).

        Args:
            x: (B, C, D, H, W)

        Returns:
            features: (B, last_hidden_dim) — e.g., (B, 128) for default config.
        """
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        for layer in list(self.mlp.children())[:-1]:
            x = layer(x)

        return x
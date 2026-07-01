"""
FC MLP Baseline Model
Simple MLP classifier: flatten FC matrix → MLP hidden layers → binary classification.

This is the simplest baseline for brain network classification — no attention,
no graph structure, just raw FC values fed through fully-connected layers.

Input:  FC matrix of shape (B, num_rois, num_rois)
Output: Binary classification logits of shape (B, num_classes)
"""

import torch
import torch.nn as nn
from typing import Optional, List


class FCMlpClassifier(nn.Module):
    """
    Simple MLP baseline for FC-based brain network classification.

    Architecture:
        FC Matrix (B, N, N) → Flatten (B, N*N) → MLP → logits (B, num_classes)
    """

    def __init__(
        self,
        num_rois: int = 116,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
        activation: nn.Module = nn.ReLU,
        use_batch_norm: bool = True,
        **kwargs
    ):
        """
        Args:
            num_rois: Number of ROIs in the FC matrix (e.g., 116 for AAL).
            hidden_dims: List of hidden layer dimensions. If None, uses [512, 128].
            num_classes: Number of output classes (2 for binary).
            dropout: Dropout rate applied after each hidden layer.
            activation: Activation function class.
            use_batch_norm: Whether to apply BatchNorm1d after each hidden Linear.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.num_rois = num_rois
        self.num_classes = num_classes

        input_dim = num_rois * num_rois  # e.g., 116*116 = 13456

        if hidden_dims is None:
            hidden_dims = [512, 128]

        # Build MLP layers
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
        """Initialize weights with truncated normal for better convergence."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fc_matrix: (B, num_rois, num_rois) — functional connectivity matrix.

        Returns:
            logits: (B, num_classes) — raw classification scores.
        """
        B = fc_matrix.shape[0]
        x = fc_matrix.view(B, -1)       # (B, N*N)
        x = self.mlp(x)                  # (B, num_classes)
        return x

    def extract_features(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the penultimate layer (before the final Linear).

        Args:
            fc_matrix: (B, num_rois, num_rois)

        Returns:
            features: (B, last_hidden_dim) — e.g., (B, 128) for default config.
        """
        B = fc_matrix.shape[0]
        x = fc_matrix.view(B, -1)

        # Apply all layers except the last (classification head)
        for layer in list(self.mlp.children())[:-1]:
            x = layer(x)

        return x
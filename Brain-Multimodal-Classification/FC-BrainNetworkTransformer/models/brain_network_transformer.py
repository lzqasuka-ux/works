"""
Classic BrainNetworkTransformer for Functional Connectivity (FC) Based Brain Network Classification.

Reference: BrainNetworkTransformer treats each ROI as a token and applies
Transformer self-attention across brain regions to capture topological patterns
in functional connectivity networks.

Input:  FC matrix of shape (B, num_rois, num_rois)
Output: Binary classification logits of shape (B, num_classes)

NOTE: Graph-specific components (e.g., edge features, graph convolutions)
are marked with TODO for later implementation if needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ==============================================================================
# Basic Building Blocks
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention (MSA) for ROI token sequences.

    Input:  (B, N, d_model)
    Output: (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)                                  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale      # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = attn @ v                                       # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)             # (B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForwardNetwork(nn.Module):
    """Two-layer MLP with GELU activation and Dropout."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer Encoder layer: MSA → Residual + LN → FFN → Residual + LN
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.msa = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            hidden_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSA sub-block
        x = x + self.msa(self.norm1(x))
        # FFN sub-block
        x = x + self.ffn(self.norm2(x))
        return x


# ==============================================================================
# ROI Embedding Layer
# ==============================================================================

class ROIEmbedding(nn.Module):
    """
    Convert each ROI's FC profile into a token embedding.

    The FC matrix (num_rois, num_rois) is treated as num_rois tokens,
    each with an initial feature vector of length num_rois (its connectivity
    to all other ROIs).

    Input:  (B, num_rois, num_rois)  — raw FC matrix
    Output: (B, num_rois, d_model)   — ROI token embeddings
    """

    def __init__(
        self,
        num_rois: int,
        d_model: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            num_rois: Number of ROIs (e.g., 116 for AAL atlas).
            d_model: Output embedding dimension.
            dropout: Dropout rate after projection.
        """
        super().__init__()
        self.num_rois = num_rois
        self.d_model = d_model

        # Project each ROI's FC profile (num_rois dim) → d_model
        self.proj = nn.Linear(num_rois, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fc_matrix: (B, num_rois, num_rois) — FC matrix.

        Returns:
            (B, num_rois, d_model) — embedded ROI tokens.
        """
        x = self.proj(fc_matrix)     # (B, num_rois, d_model)
        x = self.norm(x)
        x = self.dropout(x)
        return x


# ==============================================================================
# Positional Encoding for ROIs
# ==============================================================================

class ROIPositionalEncoding(nn.Module):
    """
    Learnable positional encoding for ROI tokens.

    Since brain ROIs have a fixed ordering (e.g., by atlas), each ROI index
    gets a learnable positional embedding added to its token.
    """

    def __init__(self, num_rois: int, d_model: int):
        """
        Args:
            num_rois: Number of ROI positions.
            d_model: Embedding dimension.
        """
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_rois, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_rois, d_model)

        Returns:
            (B, num_rois, d_model) — tokens with positional encoding added.
        """
        return x + self.pos_embed


# ==============================================================================
# BrainNetworkTransformer Backbone
# ==============================================================================

class BrainNetworkTransformer(nn.Module):
    """
    Transformer Encoder backbone for brain FC network analysis.

    Architecture:
        FC Matrix → ROIEmbedding → + PosEncoding → N× TransformerEncoderLayer → output tokens

    Output: (B, num_rois, d_model) — contextualized ROI representations.
    """

    def __init__(
        self,
        num_rois: int = 116,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ):
        """
        Args:
            num_rois: Number of brain ROIs (tokens).
            d_model: Token embedding dimension.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            ffn_dim: Hidden dimension of FFN (default: 4 * d_model).
            dropout: Dropout rate.
            qkv_bias: Whether QKV projection has bias.
        """
        super().__init__()
        self.num_rois = num_rois
        self.d_model = d_model

        # ROI embedding (FC profile → token)
        self.roi_embed = ROIEmbedding(
            num_rois=num_rois,
            d_model=d_model,
            dropout=dropout,
        )

        # Learnable ROI positional encoding
        self.pos_encoding = ROIPositionalEncoding(
            num_rois=num_rois,
            d_model=d_model,
        )

        # Class token (optional [CLS] token for classification)
        # TODO: If you prefer [CLS] token aggregation instead of mean pooling,
        # uncomment the following and adjust forward() and head accordingly.
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                qkv_bias=qkv_bias,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fc_matrix: (B, num_rois, num_rois) — FC matrix.

        Returns:
            (B, num_rois, d_model) — contextualized ROI token representations.
        """
        # Embed each ROI's connectivity profile
        x = self.roi_embed(fc_matrix)         # (B, num_rois, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)                       # (B, num_rois, d_model)
        return x


# ==============================================================================
# Classification Model (Backbone + Readout → Classification)
# ==============================================================================

class FCBrainNetworkClassifier(nn.Module):
    """
    BrainNetworkTransformer for FC-based brain network classification.

    Wraps BrainNetworkTransformer backbone with a readout + classification head:
        FC Matrix → BrainNetworkTransformer → Mean Pooling → Linear → logits

    Input:  (B, num_rois, num_rois)  — functional connectivity matrix
    Output: (B, num_classes)          — classification logits (default: 2)
    """

    def __init__(
        self,
        num_classes: int = 2,
        num_rois: int = 116,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        pool_method: str = "mean",
        **kwargs
    ):
        """
        Args:
            num_classes: Number of output classes (2 for binary).
            num_rois: Number of brain ROIs (e.g., 116 for AAL, 90 for Automated).
            d_model: Token embedding dimension.
            num_layers: Number of Transformer encoder layers.
            num_heads: Number of attention heads.
            ffn_dim: FFN hidden dimension (default: 4 * d_model).
            dropout: Dropout rate throughout the network.
            qkv_bias: Whether QKV linear projections include bias.
            pool_method: Token aggregation method: 'mean' or 'cls'.
                         'mean' averages all ROI tokens.
                         'cls' uses a prepended [CLS] token (TODO: implement if needed).
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.pool_method = pool_method

        # Backbone
        self.backbone = BrainNetworkTransformer(
            num_rois=num_rois,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )

        # Classification head
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize classification head weights."""
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.0)

    def forward(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fc_matrix: (B, num_rois, num_rois) — functional connectivity matrix.

        Returns:
            logits: (B, num_classes) — raw classification scores.
        """
        # Backbone: ROI token encoding
        x = self.backbone(fc_matrix)             # (B, num_rois, d_model)

        # Readout: aggregate ROI tokens
        if self.pool_method == "mean":
            x = x.mean(dim=1)                    # (B, d_model) — mean pooling
        elif self.pool_method == "cls":
            # TODO: If [CLS] token is used, extract cls_token output here.
            # x = x[:, 0]  # first token = [CLS]
            raise NotImplementedError(
                "CLS token pooling not yet implemented. "
                "Uncomment cls_token in BrainNetworkTransformer and adjust forward."
            )
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")

        # Classification head
        x = self.head_norm(x)
        x = self.head(x)                          # (B, num_classes)
        return x

    def extract_features(self, fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classification head.

        Args:
            fc_matrix: (B, num_rois, num_rois)

        Returns:
            features: (B, d_model)
        """
        x = self.backbone(fc_matrix)
        if self.pool_method == "mean":
            x = x.mean(dim=1)
        return x

    def get_attention_maps(self, fc_matrix: torch.Tensor) -> list:
        """
        Extract attention maps from encoder layers (for interpretability).

        NOTE: This is a placeholder. To actually extract attention weights,
        you need to modify the MSA forward to return attention matrices.

        Args:
            fc_matrix: (B, num_rois, num_rois)

        Returns:
            Empty list (placeholder).
        """
        # TODO: Implement attention map extraction by modifying
        # MultiHeadSelfAttention.forward to optionally return attn weights.
        return []
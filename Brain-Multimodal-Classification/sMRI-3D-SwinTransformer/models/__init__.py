"""
sMRI-3D-SwinTransformer Models Module

Provides the 3D Swin Transformer model for sMRI brain image classification.
"""

from .swin_transformer_3d import (
    sMRI3DSwinTransformer,
    SwinTransformer3D,
)

__all__ = [
    "sMRI3DSwinTransformer",
    "SwinTransformer3D",
]
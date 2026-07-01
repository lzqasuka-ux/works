"""
sMRI-3D-SwinTransformer Dataset Module

Provides sMRI3DDataset and dataloader utilities for
3D structural MRI Swin Transformer classification training.
"""

from .dataset import (
    sMRI3DDataset,
    get_dataloader,
    create_dataloaders,
)

__all__ = [
    "sMRI3DDataset",
    "get_dataloader",
    "create_dataloaders",
]
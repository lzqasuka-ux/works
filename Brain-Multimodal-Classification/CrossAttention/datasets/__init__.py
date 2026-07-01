"""
CrossAttention Dataset Module

Provides CrossAttentionDataset and dataloader utilities for
multi-modal cross-attention brain classification training.
"""

from .dataset import (
    CrossAttentionDataset,
    get_dataloader,
    create_dataloaders,
)

__all__ = [
    "CrossAttentionDataset",
    "get_dataloader",
    "create_dataloaders",
]
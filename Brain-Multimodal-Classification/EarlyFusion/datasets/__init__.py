"""
EarlyFusion Dataset Module

Provides EarlyFusionDataset and dataloader utilities for
early-fusion multimodal brain classification training.
"""

from .dataset import (
    EarlyFusionDataset,
    get_dataloader,
    create_dataloaders,
)

__all__ = [
    "EarlyFusionDataset",
    "get_dataloader",
    "create_dataloaders",
]
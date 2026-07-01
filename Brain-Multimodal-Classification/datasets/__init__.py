"""
Brain Multimodal Classification - Shared Dataset Module

Provides base dataset classes and utilities for loading brain imaging datasets
(ABIDE, ADHD200, ADNI, COBRE) with standardized interfaces for deep learning training.
"""

from .base_dataset import (
    BaseBrainDataset,
    get_dataloader,
    create_dataloaders,
)

__all__ = [
    "BaseBrainDataset",
    "get_dataloader",
    "create_dataloaders",
]
"""
Baseline Dataset Module

Provides baseline dataset classes and dataloader utilities for:
  - FC-based MLP classification (BaselineDataset)
  - sMRI-based MLP classification (sMRIBaselineDataset)
Both support the standard 4-dataset interface.
"""

from .dataset import (
    BaselineDataset,
    get_dataloader,
    create_dataloaders,
)
from .smri_dataset import (
    sMRIBaselineDataset,
    get_smri_dataloader,
    create_smri_dataloaders,
)

__all__ = [
    "BaselineDataset",
    "get_dataloader",
    "create_dataloaders",
    "sMRIBaselineDataset",
    "get_smri_dataloader",
    "create_smri_dataloaders",
]
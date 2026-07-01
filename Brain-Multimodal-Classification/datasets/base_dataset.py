"""
Base Brain Dataset Module
Provides common dataset loading utilities for brain multimodal classification.
Supports loading 4 datasets with a unified interface.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np


class BaseBrainDataset(Dataset):
    """
    Base class for brain imaging datasets.
    
    Supports up to 4 datasets loaded from separate paths, with configurable
    preprocessing, train/val/test splitting, and standardized interface for
    multimodal brain classification tasks.
    """

    def __init__(
        self,
        dataset1_path: Optional[str] = None,
        dataset2_path: Optional[str] = None,
        dataset3_path: Optional[str] = None,
        dataset4_path: Optional[str] = None,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            dataset1_path: Path to first dataset (e.g., ABIDE).
            dataset2_path: Path to second dataset (e.g., ADHD200).
            dataset3_path: Path to third dataset (e.g., ADNI).
            dataset4_path: Path to fourth dataset (e.g., COBRE).
            split: One of 'train', 'val', 'test', or 'all'.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            seed: Random seed for reproducible splitting.
            transform: Optional transform applied to input data.
            target_transform: Optional transform applied to labels.
            **kwargs: Additional dataset-specific arguments.
        """
        super().__init__()
        self.dataset_paths = {
            "dataset1": dataset1_path,
            "dataset2": dataset2_path,
            "dataset3": dataset3_path,
            "dataset4": dataset4_path,
        }
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

        self._data = []
        self._labels = []
        self._indices = []

        self._load_data()
        self._apply_split()

    def _load_data(self):
        """
        Load data from all 4 dataset paths.
        Override this method in subclasses to implement dataset-specific loading logic.
        """
        raise NotImplementedError(
            "Subclasses must implement `_load_data` to load data from dataset paths."
        )

    def _apply_split(self):
        """
        Apply train/val/test split to the loaded data.
        Uses deterministic splitting based on the seed.
        """
        if self.split == "all" or len(self._data) == 0:
            self._indices = list(range(len(self._data)))
            return

        n = len(self._data)
        indices = list(range(n))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)

        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        if self.split == "train":
            self._indices = indices[:train_end]
        elif self.split == "val":
            self._indices = indices[train_end:val_end]
        elif self.split == "test":
            self._indices = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}. Expected 'train', 'val', 'test', or 'all'.")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Index of the sample to fetch.

        Returns:
            Tuple of (data, label) tensors.
        """
        real_idx = self._indices[idx]
        sample = self._data[real_idx]
        label = self._labels[real_idx]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label

    def get_num_classes(self) -> int:
        """Return the number of unique classes in the dataset."""
        unique_labels = set()
        for idx in self._indices:
            label = self._labels[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            unique_labels.add(label)
        return len(unique_labels)

    @property
    def data_shape(self) -> Tuple:
        """Return the shape of a single data sample."""
        if len(self._data) > 0:
            sample = self._data[0]
            return tuple(sample.shape) if hasattr(sample, "shape") else ()
        return ()

    @property
    def num_datasets_loaded(self) -> int:
        """Return how many of the 4 dataset paths are non-empty."""
        return sum(1 for p in self.dataset_paths.values() if p is not None)


def get_dataloader(
    dataset1_path: Optional[str] = None,
    dataset2_path: Optional[str] = None,
    dataset3_path: Optional[str] = None,
    dataset4_path: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    split: str = "train",
    dataset_cls: type = BaseBrainDataset,
    **dataset_kwargs
) -> DataLoader:
    """
    Factory function to create a DataLoader with 4 dataset interface.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size for training/evaluation.
        shuffle: Whether to shuffle the data.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        split: One of 'train', 'val', 'test', or 'all'.
        dataset_cls: Dataset class to instantiate (default: BaseBrainDataset).
        **dataset_kwargs: Additional keyword arguments passed to the dataset class.

    Returns:
        DataLoader instance configured with the specified dataset.
    """
    dataset = dataset_cls(
        dataset1_path=dataset1_path,
        dataset2_path=dataset2_path,
        dataset3_path=dataset3_path,
        dataset4_path=dataset4_path,
        split=split,
        **dataset_kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if split == "train" else False),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )

    return dataloader


def create_dataloaders(
    dataset1_path: Optional[str] = None,
    dataset2_path: Optional[str] = None,
    dataset3_path: Optional[str] = None,
    dataset4_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    dataset_cls: type = BaseBrainDataset,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders with 4 dataset interface.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size.
        num_workers: Number of data loading subprocesses.
        pin_memory: Whether to pin memory for faster GPU transfer.
        dataset_cls: Dataset class to instantiate.
        **dataset_kwargs: Additional arguments for the dataset class.

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to DataLoader instances.
    """
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataloaders[split] = get_dataloader(
            dataset1_path=dataset1_path,
            dataset2_path=dataset2_path,
            dataset3_path=dataset3_path,
            dataset4_path=dataset4_path,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            split=split,
            dataset_cls=dataset_cls,
            **dataset_kwargs,
        )
    return dataloaders
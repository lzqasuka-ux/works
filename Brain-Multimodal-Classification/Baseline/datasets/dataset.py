"""
Baseline Dataset Module
Simple FC-matrix dataset for MLP-based brain network classification.
Supports loading 4 datasets with functional connectivity matrices.
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, Callable

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets.base_dataset import BaseBrainDataset


class BaselineDataset(BaseBrainDataset):
    """
    Simple dataset for MLP baseline brain network classification.

    Loads functional connectivity (FC) matrices from up to 4 datasets.
    Data is returned as raw (num_rois, num_rois) matrices suitable for
    flattening and feeding into an MLP classifier.
    """

    def __init__(
        self,
        dataset1_path: Optional[str] = None,
        dataset2_path: Optional[str] = None,
        dataset3_path: Optional[str] = None,
        dataset4_path: Optional[str] = None,
        split: str = "train",
        num_rois: int = 116,
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
            num_rois: Number of ROIs in the FC matrix (e.g., 116 for AAL).
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            seed: Random seed for reproducible splitting.
            transform: Optional transform applied to FC matrices.
            target_transform: Optional transform applied to labels.
            **kwargs: Additional arguments.
        """
        self.num_rois = num_rois
        super().__init__(
            dataset1_path=dataset1_path,
            dataset2_path=dataset2_path,
            dataset3_path=dataset3_path,
            dataset4_path=dataset4_path,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            transform=transform,
            target_transform=target_transform,
            **kwargs
        )

    def _load_data(self):
        """
        Load FC matrices from all 4 dataset paths.
        Each dataset is expected to contain FC matrices of shape (num_rois, num_rois).
        """
        for key, path in self.dataset_paths.items():
            if path is None:
                continue
            if not os.path.exists(path):
                print(f"[BaselineDataset] Warning: Path not found: {path}")
                continue
            self._load_single_dataset(key, path)

    def _load_single_dataset(self, dataset_key: str, dataset_path: str):
        """
        Load a single dataset from the given path.

        Expected data format per sample: FC matrix of shape (num_rois, num_rois) + label.

        Args:
            dataset_key: Key identifying which dataset (dataset1~dataset4).
            dataset_path: Path to the dataset directory.
        """
        # TODO: Implement actual data loading from .npy, .pt, .mat files.
        # Example:
        # for subject_dir in os.listdir(dataset_path):
        #     fc = torch.tensor(np.load(os.path.join(dataset_path, subject_dir, "fc_matrix.npy")))
        #     label = load_label(os.path.join(dataset_path, subject_dir, "label.txt"))
        #     self._data.append(fc)
        #     self._labels.append(label)
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (fc_matrix, label).
            fc_matrix shape: (num_rois, num_rois)
        """
        real_idx = self._indices[idx]
        fc_matrix = self._data[real_idx]
        label = self._labels[real_idx]

        if self.transform is not None:
            fc_matrix = self.transform(fc_matrix)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return fc_matrix, label


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
    **dataset_kwargs
) -> DataLoader:
    """
    Factory function to create a DataLoader for baseline MLP training.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of subprocesses.
        pin_memory: Pin memory for faster GPU transfer.
        split: 'train', 'val', 'test', or 'all'.
        **dataset_kwargs: Additional args passed to BaselineDataset.

    Returns:
        DataLoader for MLP baseline training.
    """
    dataset = BaselineDataset(
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
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders for baseline training.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size.
        num_workers: Number of subprocesses.
        pin_memory: Pin memory for faster GPU transfer.
        **dataset_kwargs: Additional args passed to BaselineDataset.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
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
            **dataset_kwargs,
        )
    return dataloaders
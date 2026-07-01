"""
FC-BrainNetworkTransformer Dataset Module
Functional connectivity brain network dataset for transformer-based classification.
Supports loading 4 datasets with functional connectivity matrices and graph-based features.
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, Callable

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets.base_dataset import BaseBrainDataset


class FCBrainNetworkDataset(BaseBrainDataset):
    """
    Dataset for FC-BrainNetworkTransformer based brain network classification.

    Designed to load functional connectivity (FC) matrices and/or graph-structured
    brain network data from up to 4 datasets, suitable for transformer-based
    brain network analysis.
    """

    def __init__(
        self,
        dataset1_path: Optional[str] = None,
        dataset2_path: Optional[str] = None,
        dataset3_path: Optional[str] = None,
        dataset4_path: Optional[str] = None,
        split: str = "train",
        fc_matrix_key: str = "fc_matrix",
        num_rois: int = 116,
        use_correlation: bool = True,
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
            fc_matrix_key: Key/name for the FC matrix file (e.g., 'fc_matrix').
            num_rois: Number of ROIs (regions of interest) in the brain atlas.
            use_correlation: Whether to convert raw time-series to correlation matrices.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            seed: Random seed for reproducible splitting.
            transform: Optional transform applied to FC matrices.
            target_transform: Optional transform applied to labels.
            **kwargs: Additional arguments.
        """
        self.fc_matrix_key = fc_matrix_key
        self.num_rois = num_rois
        self.use_correlation = use_correlation
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
        Load functional connectivity data from all 4 dataset paths.
        Each dataset is expected to contain FC matrices (num_rois x num_rois) per subject.
        """
        for key, path in self.dataset_paths.items():
            if path is None:
                continue
            if not os.path.exists(path):
                print(f"[FCBrainNetworkDataset] Warning: Path not found: {path}")
                continue
            self._load_single_dataset(key, path)

    def _load_single_dataset(self, dataset_key: str, dataset_path: str):
        """
        Load a single dataset from the given path.

        Expected data format per sample: FC matrix of shape (num_rois, num_rois) + label.
        If use_correlation=True and raw time-series is provided, compute correlation matrix.

        Args:
            dataset_key: Key identifying which dataset (dataset1~dataset4).
            dataset_path: Path to the dataset directory.
        """
        # TODO: Implement actual data loading from brain imaging files (e.g., .npy, .pt, .mat).
        # Placeholder: iterate through subject directories and load FC matrices.
        # Example:
        # for subject_dir in os.listdir(dataset_path):
        #     if self.use_correlation:
        #         ts = load_time_series(os.path.join(dataset_path, subject_dir, "timeseries.npy"))
        #         fc_matrix = torch.corrcoef(torch.tensor(ts).T)  # (num_rois, num_rois)
        #     else:
        #         fc_matrix = torch.tensor(np.load(os.path.join(dataset_path, subject_dir, f"{self.fc_matrix_key}.npy")))
        #     label = load_label(os.path.join(dataset_path, subject_dir, "label.txt"))
        #     self._data.append(fc_matrix)
        #     self._labels.append(label)
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (fc_matrix, label).
            fc_matrix shape: (num_rois, num_rois) representing brain functional connectivity.
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
    Factory function to create a DataLoader for FC-BrainNetworkTransformer training.

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
        **dataset_kwargs: Additional args passed to FCBrainNetworkDataset.

    Returns:
        DataLoader for FC-brain network training.
    """
    dataset = FCBrainNetworkDataset(
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
    Create train, validation, and test DataLoaders for FC-BrainNetworkTransformer.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size.
        num_workers: Number of subprocesses.
        pin_memory: Pin memory for faster GPU transfer.
        **dataset_kwargs: Additional args passed to FCBrainNetworkDataset.

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
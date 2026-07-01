"""
EarlyFusion Dataset Module
Early fusion multimodal dataset for brain image classification.
Supports loading 4 datasets with early concatenation/fusion of multiple modalities.
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Callable

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets.base_dataset import BaseBrainDataset


class EarlyFusionDataset(BaseBrainDataset):
    """
    Dataset for Early-Fusion based multimodal brain classification.

    Designed to load multiple modalities of brain imaging data (e.g., fMRI, sMRI, DTI,
    clinical scores) from up to 4 datasets and concatenate them at the input level
    for early fusion training.
    """

    def __init__(
        self,
        dataset1_path: Optional[str] = None,
        dataset2_path: Optional[str] = None,
        dataset3_path: Optional[str] = None,
        dataset4_path: Optional[str] = None,
        split: str = "train",
        modality_keys: Optional[List[str]] = None,
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
            modality_keys: List of modality keys to load (e.g., ['fmri', 'smri', 'dti']).
                If None, loads all available modalities.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            seed: Random seed for reproducible splitting.
            transform: Optional transform applied to concatenated input data.
            target_transform: Optional transform applied to labels.
            **kwargs: Additional arguments.
        """
        self.modality_keys = modality_keys or ["fmri", "smri"]
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
        Load multimodal data from all 4 dataset paths.
        Multiple modalities are loaded and concatenated into a single tensor per sample.
        """
        for key, path in self.dataset_paths.items():
            if path is None:
                continue
            if not os.path.exists(path):
                print(f"[EarlyFusionDataset] Warning: Path not found: {path}")
                continue
            self._load_single_dataset(key, path)

    def _load_single_dataset(self, dataset_key: str, dataset_path: str):
        """
        Load a single dataset from the given path.

        Expected data format per sample: list of modality tensors + label.
        All modality tensors are concatenated into a single flattened feature vector
        for early fusion.

        Args:
            dataset_key: Key identifying which dataset (dataset1~dataset4).
            dataset_path: Path to the dataset directory.
        """
        # TODO: Implement actual data loading from brain imaging files (e.g., .nii, .npy, .pt).
        # Placeholder: iterate through subject directories and load all modalities.
        # Example:
        # for subject_dir in os.listdir(dataset_path):
        #     modalities = []
        #     for mkey in self.modality_keys:
        #         mod = load_modality(os.path.join(dataset_path, subject_dir, f"{mkey}.npy"))
        #         modalities.append(torch.tensor(mod).flatten())
        #     fused = torch.cat(modalities, dim=0)  # early fusion: concatenate at input level
        #     label = load_label(os.path.join(dataset_path, subject_dir, "label.txt"))
        #     self._data.append(fused)
        #     self._labels.append(label)
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (fused_features, label) for early fusion training.
            fused_features is the concatenated multi-modal tensor.
        """
        real_idx = self._indices[idx]
        fused_features = self._data[real_idx]
        label = self._labels[real_idx]

        if self.transform is not None:
            fused_features = self.transform(fused_features)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return fused_features, label


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
    Factory function to create a DataLoader for EarlyFusion training.

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
        **dataset_kwargs: Additional args passed to EarlyFusionDataset.

    Returns:
        DataLoader for early fusion training.
    """
    dataset = EarlyFusionDataset(
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
    Create train, validation, and test DataLoaders for EarlyFusion.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size.
        num_workers: Number of subprocesses.
        pin_memory: Pin memory for faster GPU transfer.
        **dataset_kwargs: Additional args passed to EarlyFusionDataset.

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
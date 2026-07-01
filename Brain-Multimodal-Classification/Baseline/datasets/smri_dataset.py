"""
Baseline sMRI Dataset Module
3D structural MRI dataset for MLP-based brain classification baseline.
Supports loading 4 datasets with 3D brain volumes.
"""

import os
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, Callable

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasets.base_dataset import BaseBrainDataset


class sMRIBaselineDataset(BaseBrainDataset):
    """
    Dataset for sMRI MLP baseline classification.

    Loads 3D structural MRI volumes from up to 4 datasets.
    Data is returned as (1, D, H, W) 3D volumes suitable for
    downsampling and feeding into an MLP classifier.
    """

    def __init__(
        self,
        dataset1_path: Optional[str] = None,
        dataset2_path: Optional[str] = None,
        dataset3_path: Optional[str] = None,
        dataset4_path: Optional[str] = None,
        split: str = "train",
        volume_shape: Tuple[int, int, int] = (96, 96, 96),
        normalize: bool = True,
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
            volume_shape: Target spatial shape for 3D volumes (D, H, W).
            normalize: Whether to apply z-score normalization per volume.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            test_ratio: Proportion of data for testing.
            seed: Random seed for reproducible splitting.
            transform: Optional transform applied to 3D volumes.
            target_transform: Optional transform applied to labels.
            **kwargs: Additional arguments.
        """
        self.volume_shape = volume_shape
        self.normalize = normalize
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
        Load 3D sMRI volumes from all 4 dataset paths.
        Each dataset is expected to contain 3D brain volumes of shape (D, H, W).
        """
        for key, path in self.dataset_paths.items():
            if path is None:
                continue
            if not os.path.exists(path):
                print(f"[sMRIBaselineDataset] Warning: Path not found: {path}")
                continue
            self._load_single_dataset(key, path)

    def _load_single_dataset(self, dataset_key: str, dataset_path: str):
        """
        Load a single dataset from the given path.

        Expected data format per sample: 3D volume of shape (D, H, W) + label.

        Args:
            dataset_key: Key identifying which dataset (dataset1~dataset4).
            dataset_path: Path to the dataset directory.
        """
        # TODO: Implement actual data loading from NIfTI (.nii/.nii.gz) or .pt/.npy files.
        # Example:
        # import nibabel as nib
        # for subject_dir in os.listdir(dataset_path):
        #     nii_path = os.path.join(dataset_path, subject_dir, "T1w.nii.gz")
        #     img = nib.load(nii_path)
        #     volume = torch.tensor(img.get_fdata(), dtype=torch.float32)
        #     if self.normalize:
        #         volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        #     volume = torch.nn.functional.interpolate(
        #         volume.unsqueeze(0).unsqueeze(0),
        #         size=self.volume_shape, mode='trilinear'
        #     ).squeeze()
        #     label = load_label(os.path.join(dataset_path, subject_dir, "label.txt"))
        #     self._data.append(volume)
        #     self._labels.append(label)
        pass

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (volume, label).
            volume shape: (1, D, H, W) — channel-first 3D tensor.
        """
        real_idx = self._indices[idx]
        volume = self._data[real_idx]
        label = self._labels[real_idx]

        # Add channel dimension if not present: (D, H, W) → (1, D, H, W)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)

        if self.transform is not None:
            volume = self.transform(volume)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return volume, label


def get_smri_dataloader(
    dataset1_path: Optional[str] = None,
    dataset2_path: Optional[str] = None,
    dataset3_path: Optional[str] = None,
    dataset4_path: Optional[str] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    split: str = "train",
    **dataset_kwargs
) -> DataLoader:
    """
    Factory function to create a DataLoader for sMRI MLP baseline training.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size (smaller default for 3D volumes).
        shuffle: Whether to shuffle.
        num_workers: Number of subprocesses.
        pin_memory: Pin memory for faster GPU transfer.
        split: 'train', 'val', 'test', or 'all'.
        **dataset_kwargs: Additional args passed to sMRIBaselineDataset.

    Returns:
        DataLoader for sMRI MLP baseline training.
    """
    dataset = sMRIBaselineDataset(
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


def create_smri_dataloaders(
    dataset1_path: Optional[str] = None,
    dataset2_path: Optional[str] = None,
    dataset3_path: Optional[str] = None,
    dataset4_path: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders for sMRI MLP baseline.

    Args:
        dataset1_path: Path to first dataset.
        dataset2_path: Path to second dataset.
        dataset3_path: Path to third dataset.
        dataset4_path: Path to fourth dataset.
        batch_size: Batch size.
        num_workers: Number of subprocesses.
        pin_memory: Pin memory for faster GPU transfer.
        **dataset_kwargs: Additional args passed to sMRIBaselineDataset.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataloaders[split] = get_smri_dataloader(
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
"""
COBRE 多模态 PyTorch Dataset
支持 sMRI (.nii.gz) + 功能连接矩阵 (FC, .csv) 的同步加载
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Union


def _fisher_z_transform(matrix: np.ndarray) -> np.ndarray:
    """对相关矩阵做 Fisher Z 变换，处理边界值。"""
    matrix = matrix.copy()
    bounds = 0.9999
    matrix = np.clip(matrix, -bounds, bounds)
    return np.arctanh(matrix)


class COBREDataset(Dataset):
    """
    COBRE 多模态数据集。

    返回每个样本的字典:
        {
            "id": str,             受试者 ID
            "sMRI": Tensor,       (1, D, H, W) 的结构 MRI
            "FC": Tensor,         (N_ROI, N_ROI) 功能连接矩阵
            "label": LongTensor,  (1,) 标签 0=Control, 1=Patient
            "sex": int,           0=女, 1=男
            "age": float,         年龄
        }

    Args:
        csv_path:      COBRE_multimodal.csv 路径
        base_dir:      数据集根目录 (相对于该目录解析 sMRI_path / FC_path)
        split:         "train" / "val" / "test" / "all"
        seed:          随机种子
        train_ratio:   训练集占比 (默认 0.7)
        val_ratio:     验证集占比 (默认 0.15), 测试集 = 1 - train - val
        fisher_z:      是否对 FC 做 Fisher Z 变换
        normalize_smri:是否将 sMRI 归一化到 [0, 1]
    """

    def __init__(
        self,
        csv_path: str = None,
        base_dir: str = None,
        split: str = "train",
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        fisher_z: bool = True,
        normalize_smri: bool = True,
    ):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(__file__), "COBRE_multimodal.csv")
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir
        self.fisher_z = fisher_z
        self.normalize_smri = normalize_smri

        # 加载元数据
        df = pd.read_csv(csv_path)
        self.ids = df["ID"].values
        self.labels = df["Label"].values.astype(np.int64)
        self.sexes = df["Sex"].values.astype(np.int64)
        self.ages = df["Age"].values.astype(np.float32)
        self.smri_rel = df["sMRI_path"].values
        self.fc_rel = df["FC_path"].values

        n_total = len(self.ids)
        print(f"[COBREDataset] 总样本: {n_total}")
        print(f"  Control (0): {(self.labels == 0).sum()}, "
              f"Patient (1): {(self.labels == 1).sum()}")

        # 分层划分 train / val / test
        indices = np.arange(n_total)
        rng = np.random.default_rng(seed)

        idx_0 = indices[self.labels == 0]
        idx_1 = indices[self.labels == 1]
        rng.shuffle(idx_0)
        rng.shuffle(idx_1)

        def _split(idx):
            n = len(idx)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            return idx[:n_train], idx[n_train: n_train + n_val], idx[n_train + n_val:]

        train_0, val_0, test_0 = _split(idx_0)
        train_1, val_1, test_1 = _split(idx_1)

        self.split_map = {
            "train": np.sort(np.concatenate([train_0, train_1])),
            "val": np.sort(np.concatenate([val_0, val_1])),
            "test": np.sort(np.concatenate([test_0, test_1])),
            "all": indices,
        }
        self.indices = self.split_map[split]

        for k, v in self.split_map.items():
            if k != "all":
                print(f"  {k}: {len(v)}")

    def __len__(self) -> int:
        return len(self.indices)

    def _load_smri(self, rel_path: str) -> np.ndarray:
        """加载 .nii.gz 返回 float32 (D, H, W)。"""
        full_path = os.path.join(self.base_dir, rel_path)
        img = nib.load(full_path)
        data = img.get_fdata(dtype=np.float32)
        if self.normalize_smri:
            mn, mx = data.min(), data.max()
            if mx > mn:
                data = (data - mn) / (mx - mn)
        return data

    def _load_fc(self, rel_path: str) -> np.ndarray:
        """加载 400×400 功能连接矩阵，跳过首行 + 每行首列。"""
        full_path = os.path.join(self.base_dir, rel_path)
        fc = np.loadtxt(full_path, delimiter=",", skiprows=1, usecols=range(1, 401))
        if self.fisher_z:
            fc = _fisher_z_transform(fc)
        return fc.astype(np.float32)

    def get_item_by_global(self, global_idx: int) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        """按全局索引取样本（内部使用）。"""
        subj_id = self.ids[global_idx]
        smri = self._load_smri(self.smri_rel[global_idx])
        fc = self._load_fc(self.fc_rel[global_idx])
        return {
            "id": subj_id,
            "sMRI": torch.from_numpy(smri).unsqueeze(0),
            "FC": torch.from_numpy(fc),
            "label": torch.tensor(self.labels[global_idx], dtype=torch.long),
            "sex": self.sexes[global_idx],
            "age": self.ages[global_idx],
        }

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int, float]]:
        return self.get_item_by_global(self.indices[idx])


def get_dataloader(
    csv_path: str = None,
    base_dir: str = None,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    fisher_z: bool = True,
    normalize_smri: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    一键获取 train/val/test DataLoader.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    full_ds = COBREDataset(
        csv_path=csv_path, base_dir=base_dir, split="all",
        seed=seed, train_ratio=train_ratio, val_ratio=val_ratio,
        fisher_z=fisher_z, normalize_smri=normalize_smri,
    )

    class _Subset(Dataset):
        def __init__(self, parent, indices):
            self.parent = parent
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.parent.get_item_by_global(self.indices[idx])

    train_ds = _Subset(full_ds, full_ds.split_map["train"])
    val_ds = _Subset(full_ds, full_ds.split_map["val"])
    test_ds = _Subset(full_ds, full_ds.split_map["test"])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  COBRE Dataset 自测")
    print("=" * 55)

    ds = COBREDataset(split="train", seed=42)
    print(f"\nDataset 长度: {len(ds)}")

    sample = ds[0]
    print(f"\n样本 0:")
    print(f"  ID:       {sample['id']}")
    print(f"  sMRI:     {tuple(sample['sMRI'].shape)}")
    print(f"  FC:       {tuple(sample['FC'].shape)}")
    print(f"  Label:    {sample['label'].item()}")
    print(f"  Sex:      {sample['sex']}")
    print(f"  Age:      {sample['age']}")

    print("\n--- DataLoader 测试 ---")
    train_loader, val_loader, test_loader = get_dataloader(batch_size=4, seed=42)
    for batch in train_loader:
        print(f"  sMRI={tuple(batch['sMRI'].shape)}, "
              f"FC={tuple(batch['FC'].shape)}, "
              f"label={batch['label'].tolist()}")
        break

    print(f"\n  val loader 第一批: sMRI={next(iter(val_loader))['sMRI'].shape}")
    print(f"  test loader 第一批: sMRI={next(iter(test_loader))['sMRI'].shape}")
    print("\n  所有测试通过!")
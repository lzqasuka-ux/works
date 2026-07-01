# Brain-Multimodal-Classification 数据集加载函数说明

## 目录结构

```
Brain-Multimodal-Classification/
│
├── datasets/                              # 共享基础模块
│   ├── __init__.py                        # 导出 BaseBrainDataset, get_dataloader, create_dataloaders
│   ├── base_dataset.py                    # 基类 + 通用 DataLoader 工厂函数
│   ├── ABIDE/                             # ABIDE 数据集目录（待填充数据）
│   ├── ADHD200/                           # ADHD200 数据集目录（待填充数据）
│   ├── ADNI/                              # ADNI 数据集目录（待填充数据）
│   └── COBRE/                             # COBRE 数据集目录（待填充数据）
│
├── CrossAttention/datasets/               # CrossAttention 模型专用数据集
│   ├── __init__.py
│   └── dataset.py                         # CrossAttentionDataset
│
├── EarlyFusion/datasets/                  # EarlyFusion 模型专用数据集
│   ├── __init__.py
│   └── dataset.py                         # EarlyFusionDataset
│
├── FC-BrainNetworkTransformer/datasets/   # FC-BrainNetworkTransformer 专用数据集
│   ├── __init__.py
│   └── dataset.py                         # FCBrainNetworkDataset
│
└── sMRI-3D-SwinTransformer/datasets/      # sMRI-3D-SwinTransformer 专用数据集
    ├── __init__.py
    └── dataset.py                         # sMRI3DDataset
```

---

## 一、共享基础模块 `datasets/`

### `BaseBrainDataset`（基类）

所有模型数据集类的父类，继承自 `torch.utils.data.Dataset`。

**构造函数参数：**

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `dataset1_path` | `str \| None` | `None` | 第1个数据集路径（如 ABIDE） |
| `dataset2_path` | `str \| None` | `None` | 第2个数据集路径（如 ADHD200） |
| `dataset3_path` | `str \| None` | `None` | 第3个数据集路径（如 ADNI） |
| `dataset4_path` | `str \| None` | `None` | 第4个数据集路径（如 COBRE） |
| `split` | `str` | `"train"` | 数据集划分：`"train"` / `"val"` / `"test"` / `"all"` |
| `train_ratio` | `float` | `0.7` | 训练集比例 |
| `val_ratio` | `float` | `0.15` | 验证集比例 |
| `test_ratio` | `float` | `0.15` | 测试集比例 |
| `seed` | `int` | `42` | 随机种子（保证划分可复现） |
| `transform` | `Callable \| None` | `None` | 数据增强/预处理变换 |
| `target_transform` | `Callable \| None` | `None` | 标签变换 |

**核心属性/方法：**

| 属性/方法 | 返回类型 | 说明 |
|-----------|----------|------|
| `__len__()` | `int` | 当前 split 中的样本数量 |
| `__getitem__(idx)` | `Tuple[Tensor, Tensor]` | 返回 (data, label) |
| `get_num_classes()` | `int` | 数据集的类别数 |
| `data_shape` | `Tuple` | 单个样本的数据形状 |
| `num_datasets_loaded` | `int` | 实际加载的数据集数量（0~4） |

**需要子类实现的方法：**

| 方法 | 说明 |
|------|------|
| `_load_data()` | 从 4 个数据集路径加载数据，填充 `self._data` 和 `self._labels` |
| `_load_single_dataset(dataset_key, dataset_path)` | （可选）加载单个数据集的辅助方法 |

---

### `get_dataloader()`（通用 DataLoader 工厂函数）

```python
from datasets import get_dataloader

dataloader = get_dataloader(
    dataset1_path="/data/ABIDE",
    dataset2_path="/data/ADHD200",
    dataset3_path="/data/ADNI",
    dataset4_path="/data/COBRE",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    split="train",
    dataset_cls=BaseBrainDataset,
    **dataset_kwargs
)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset1_path` ~ `dataset4_path` | `str \| None` | `None` | 四个数据集路径 |
| `batch_size` | `int` | `32` | 批大小 |
| `shuffle` | `bool` | `True` | 是否打乱 |
| `num_workers` | `int` | `4` | 数据加载子进程数 |
| `pin_memory` | `bool` | `True` | 是否锁定内存（GPU 训练建议开启） |
| `split` | `str` | `"train"` | 数据集划分 |
| `dataset_cls` | `type` | `BaseBrainDataset` | 数据集类 |
| `**dataset_kwargs` | — | — | 传递给数据集类的额外参数 |

---

### `create_dataloaders()`（一键创建 train/val/test）

```python
from datasets import create_dataloaders

loaders = create_dataloaders(
    dataset1_path="/data/ABIDE",
    dataset2_path="/data/ADHD200",
    dataset3_path="/data/ADNI",
    dataset4_path="/data/COBRE",
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    dataset_cls=BaseBrainDataset,
    **dataset_kwargs
)

train_loader = loaders["train"]
val_loader   = loaders["val"]
test_loader  = loaders["test"]
```

---

## 二、CrossAttention 数据集

**导入方式：**
```python
from CrossAttention.datasets import CrossAttentionDataset, get_dataloader, create_dataloaders
```

**`CrossAttentionDataset` 特有参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `modality_a_key` | `str` | `"fmri"` | 模态A 标识（如 fMRI） |
| `modality_b_key` | `str` | `"smri"` | 模态B 标识（如 sMRI） |

**`__getitem__` 返回值：** `(modality_a, modality_b, label)` — 双模态输入用于交叉注意力融合。

**使用示例：**
```python
dataloader = get_dataloader(
    dataset1_path="/data/ABIDE",
    dataset2_path="/data/ADHD200",
    dataset3_path=None,
    dataset4_path=None,
    batch_size=16,
    split="train",
    modality_a_key="fmri",
    modality_b_key="smri",
)

for modality_a, modality_b, labels in dataloader:
    # modality_a: fMRI 数据
    # modality_b: sMRI 数据
    # 送入 Cross-Attention 模型
    pass
```

---

## 三、EarlyFusion 数据集

**导入方式：**
```python
from EarlyFusion.datasets import EarlyFusionDataset, get_dataloader, create_dataloaders
```

**`EarlyFusionDataset` 特有参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `modality_keys` | `List[str] \| None` | `["fmri", "smri"]` | 要加载的模态列表，传入 `None` 加载全部 |

**`__getitem__` 返回值：** `(fused_features, label)` — 多模态特征拼接后的向量。

**使用示例：**
```python
dataloader = get_dataloader(
    dataset1_path="/data/ABIDE",
    dataset2_path="/data/ADNI",
    dataset3_path="/data/COBRE",
    dataset4_path=None,
    batch_size=64,
    split="train",
    modality_keys=["fmri", "smri", "dti", "clinical"],
)
```

---

## 四、FC-BrainNetworkTransformer 数据集

**导入方式：**
```python
from FC_BrainNetworkTransformer.datasets import FCBrainNetworkDataset, get_dataloader, create_dataloaders
```

**`FCBrainNetworkDataset` 特有参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fc_matrix_key` | `str` | `"fc_matrix"` | FC 矩阵文件名标识 |
| `num_rois` | `int` | `116` | 脑区 ROI 数量（AAL 模板默认 116） |
| `use_correlation` | `bool` | `True` | 是否从时间序列计算皮尔逊相关系数矩阵 |

**`__getitem__` 返回值：** `(fc_matrix, label)` — fc_matrix 形状为 `(num_rois, num_rois)`。

**使用示例：**
```python
dataloader = get_dataloader(
    dataset1_path="/data/ABIDE",
    batch_size=32,
    split="train",
    num_rois=116,
    use_correlation=True,
)
```

---

## 五、sMRI-3D-SwinTransformer 数据集

**导入方式：**
```python
from sMRI_3D_SwinTransformer.datasets import sMRI3DDataset, get_dataloader, create_dataloaders
```

**`sMRI3DDataset` 特有参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `volume_shape` | `Tuple[int,int,int]` | `(96, 96, 96)` | 统一重采样后的 3D 体积形状 (D, H, W) |
| `patch_size` | `Tuple[int,int,int]` | `(4, 4, 4)` | 3D Swin Transformer 的 patch 大小 |
| `normalize` | `bool` | `True` | 是否做 z-score 标准化 |

**`__getitem__` 返回值：** `(volume, label)` — volume 形状为 `(1, D, H, W)`。

**属性 `num_patches`：** 返回 `(D//patch_d, H//patch_h, W//patch_w)`。

**使用示例：**
```python
dataloader = get_dataloader(
    dataset1_path="/data/ADNI",
    batch_size=4,          # 3D 体积显存占用大，建议用较小的 batch_size
    split="train",
    volume_shape=(128, 128, 128),
    patch_size=(4, 4, 4),
    normalize=True,
)
```

---

## 六、在 train.py 中的典型用法

以 `CrossAttention/train.py` 为例：

```python
"""
CrossAttention Training Script
"""
import torch
from CrossAttention.datasets import create_dataloaders

def main():
    # 1. 加载 4 个数据集
    dataloaders = create_dataloaders(
        dataset1_path="/data/ABIDE",
        dataset2_path="/data/ADHD200",
        dataset3_path="/data/ADNI",
        dataset4_path="/data/COBRE",
        batch_size=32,
        num_workers=4,
        modality_a_key="fmri",
        modality_b_key="smri",
    )

    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]
    test_loader  = dataloaders["test"]

    # 2. 获取类别数（用于模型输出层配置）
    num_classes = train_loader.dataset.get_num_classes()

    # 3. 训练循环
    for epoch in range(num_epochs):
        for modality_a, modality_b, labels in train_loader:
            # modality_a, modality_b, labels = modality_a.cuda(), ...
            # outputs = model(modality_a, modality_b)
            # loss = criterion(outputs, labels)
            pass

if __name__ == "__main__":
    main()
```

---

## 七、待实现内容

每个 `dataset.py` 中的 `_load_single_dataset()` 方法目前为占位实现（`pass`），需要根据实际数据格式填充加载逻辑。文件内已包含带注释的示例代码，支持的数据格式包括：

- `.npy` — NumPy 数组
- `.pt` / `.pth` — PyTorch 张量
- `.nii` / `.nii.gz` — NIfTI 医学影像（需 `nibabel` 库）
- `.mat` — MATLAB 文件（需 `scipy` 库）

在对应方法的 `# TODO` 处填充实际的数据读取逻辑即可。
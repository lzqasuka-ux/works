# Baseline — FC MLP 基线模型使用说明

## 模型概述

最简单的脑网络分类基线：将 FC 矩阵直接 flatten，通过几层 MLP 做二分类。**无注意力、无图结构、无卷积**，纯粹的全连接层，作为所有其他复杂模型的对比基准。

### 架构流程

```
输入: (B, num_rois, num_rois)  FC 功能连接矩阵 (如 B, 116, 116)
  │
  ├─ Flatten: (B, num_rois*num_rois)    (如 B, 13456)
  ├─ Linear(13456, 512) + BN + ReLU + Dropout(0.3)
  ├─ Linear(512, 128)   + BN + ReLU + Dropout(0.3)
  └─ Linear(128, 2)                     → logits (B, 2)
```

### 默认参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_rois` | 116 | 脑区数量（AAL 模板） |
| `hidden_dims` | `[512, 128]` | 隐藏层维度列表 |
| `dropout` | 0.3 | 每层后的 Dropout 率 |
| `use_batch_norm` | True | 是否使用 BatchNorm1d |
| `activation` | ReLU | 激活函数 |
| 参数量 | ~7M | 主要是第一层 (13456×512=6.9M) |

---

## 模型参数说明

### `FCMlpClassifier` 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_rois` | `int` | `116` | FC 矩阵中的 ROI 数量 |
| `hidden_dims` | `List[int] \| None` | `[512, 128]` | 隐藏层维度（支持任意深度） |
| `num_classes` | `int` | `2` | 分类类别数 |
| `dropout` | `float` | `0.3` | 每层隐藏层后的 Dropout 率 |
| `activation` | `nn.Module` | `nn.ReLU` | 激活函数类 |
| `use_batch_norm` | `bool` | `True` | 是否使用 BatchNorm1d |

---

## 使用方法

### 快速开始

```python
import torch
from Baseline.models.fc_mlp_baseline import FCMlpClassifier

# 1. 初始化模型
model = FCMlpClassifier(
    num_rois=116,
    hidden_dims=[512, 128],
    num_classes=2,
    dropout=0.3,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. 输入 (B, num_rois, num_rois)
fc_matrix = torch.randn(4, 116, 116).to(device)

# 3. 前向传播
logits = model(fc_matrix)           # (4, 2)
```

### 配合数据集的完整训练示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from Baseline.models import FCMlpClassifier
from Baseline.datasets import create_dataloaders

# 数据加载（4 数据集接口）
loaders = create_dataloaders(
    dataset1_path="/data/ABIDE",
    dataset2_path="/data/ADHD200",
    dataset3_path="/data/ADNI",
    dataset4_path=None,
    batch_size=64,
    num_rois=116,
)

train_loader = loaders["train"]
val_loader   = loaders["val"]
test_loader  = loaders["test"]

# 模型
model = FCMlpClassifier(num_rois=116, num_classes=2).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
model.train()
for epoch in range(50):
    for fc_matrices, labels in train_loader:
        fc_matrices = fc_matrices.cuda()
        labels = labels.cuda()

        logits = model(fc_matrices)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 自定义隐藏层深度

```python
# 浅层 MLP（更快但精度可能更低）
model = FCMlpClassifier(hidden_dims=[256])

# 深层 MLP（更多参数）
model = FCMlpClassifier(hidden_dims=[1024, 512, 256, 64])

# 不用 BatchNorm
model = FCMlpClassifier(use_batch_norm=False)
```

### 特征提取

```python
model = FCMlpClassifier(num_rois=116)
features = model.extract_features(fc_matrices)  # (B, 128) — 分类头前一层输出
```

---

## 与 BrainNetworkTransformer 的对比

| 特性 | FC-MLP (Baseline) | BrainNetworkTransformer |
|------|-------------------|------------------------|
| 结构 | FC flatten → MLP | ROI token 化 → Transformer Encoder |
| 参数量 | ~7M | ~8M |
| 脑区交互建模 | 无（第一层全连接隐式） | 显式（自注意力逐对 ROI） |
| 位置编码 | 无 | 可学习 ROI 位置编码 |
| 可解释性 | 低 | 中等（可提取注意力图） |
| 训练速度 | 快 | 较快 |

**MLP 基线的作用**：如果 BrainNetworkTransformer 的效果不比这个简单的 MLP 好多少，说明自注意力对 FC 数据可能没有额外增益；反之则证明脑区交互建模是有效的。

---

## sMRI MLP 基线（AvgPool3d + MLP）

### 模型概述

sMRI 3D 体积的极简基线：**AvgPool3d 大幅降采样 → Flatten → MLP 分类**。无卷积、无注意力，粗暴地丢弃了所有空间结构，仅靠全局平均池化后的体素强度分布来做分类。

> 设计意图：和 `sMRI-3D-SwinTransformer` 对比，衡量 Swin Transformer 的空间结构建模到底贡献了多少。

### 架构流程

```
输入: (B, 1, 96, 96, 96)  3D sMRI 脑体积 (884,736 个体素)
  │
  ├─ AvgPool3d(kernel=12)       → (B, 1, 8, 8, 8)   (512 个值)
  │                                ⚠️ 所有空间结构在此丢弃
  ├─ Flatten                    → (B, 512)
  ├─ Linear(512, 256) + BN + ReLU + Dropout(0.3)
  ├─ Linear(256, 128)  + BN + ReLU + Dropout(0.3)
  └─ Linear(128, 2)             → logits (B, 2)
```

### 默认参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `volume_shape` | `(96, 96, 96)` | 输入 3D 体积尺寸 |
| `pool_kernel` | 自动(volume/8) | AvgPool 核大小，如 96→8 得 kernel=12 |
| `hidden_dims` | `[256, 128]` | 隐藏层维度列表 |
| `dropout` | 0.3 | 每层后的 Dropout 率 |
| `use_batch_norm` | True | 是否使用 BatchNorm1d |
| `activation` | ReLU | 激活函数 |
| 参数量 | ~0.3M | 极小（512→256→128→2） |

### 与 sMRI-3D-SwinTransformer 的对比

| 特性 | sMRI-MLP (Baseline) | sMRI-3D-SwinTransformer |
|------|---------------------|-------------------------|
| 结构 | AvgPool3d + MLP | 3D Patch + Swin Attention |
| 参数量 | ~0.3M | ~40M |
| 空间结构 | 完全丢弃 | 显式建模（窗口自注意力） |
| 局部模式 | 无法捕获 | 通过 3D 窗口学习 |
| 训练速度 | 极快 | 慢（显存大） |
| 可解释性 | 无 | 中等 |

### 快速开始

```python
import torch
from Baseline.models import sMRIMlpClassifier
from Baseline.datasets import create_smri_dataloaders

# 1. 模型
model = sMRIMlpClassifier(
    volume_shape=(96, 96, 96),
    hidden_dims=[256, 128],
    num_classes=2,
)
model = model.cuda()

# 2. 数据加载（4 数据集接口）
loaders = create_smri_dataloaders(
    dataset1_path="/data/ADNI",
    dataset2_path="/data/COBRE",
    batch_size=8,
    volume_shape=(96, 96, 96),
)
train_loader = loaders["train"]

# 3. 前向传播
volume = torch.randn(8, 1, 96, 96, 96).cuda()
logits = model(volume)              # (8, 2)
features = model.extract_features(volume)  # (8, 128)
```

---

## 文件结构

```
Baseline/
├── __init__.py                   # 模块说明
├── MODEL_USAGE.md                # 本文件
├── datasets/
│   ├── __init__.py               # 导出 BaselineDataset + sMRIBaselineDataset
│   ├── dataset.py                # BaselineDataset(BaseBrainDataset) — FC 4 数据集接口
│   └── smri_dataset.py           # sMRIBaselineDataset(BaseBrainDataset) — sMRI 4 数据集接口
└── models/
    ├── __init__.py               # 导出 FCMlpClassifier + sMRIMlpClassifier
    ├── fc_mlp_baseline.py        # FCMlpClassifier 实现
    └── smri_mlp_baseline.py      # sMRIMlpClassifier 实现
```

---

## 依赖

```
torch >= 1.10
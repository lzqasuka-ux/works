# sMRI-3D-SwinTransformer 模型使用说明

## 模型概述

基于 **Video Swin Transformer (CVPR 2022)** 改编的 3D 版本，用于从结构磁共振成像（sMRI）3D 脑体积中提取特征并进行二分类（如 AD vs NC、患者 vs 对照）。

### 架构流程

```
输入: (B, 1, D, H, W)  3D sMRI 脑体积
  │
  ├─ PatchEmbed3D       → 非重叠 3D patch 划分 + 线性投影
  │                        (4, 96, 96, 96) → (24³, 96) = (13824, 96)
  │
  ├─ Stage 1            → 2× SwinBlock3D (96 dim, 24³ resolution, win=7³)
  │                       输出: (13824, 96)
  │
  ├─ PatchMerging3D     → 2×2×2 合并, 空间减半, 通道翻倍
  │                       输出: (12³, 192) = (1728, 192)
  │
  ├─ Stage 2            → 2× SwinBlock3D (192 dim, 12³ resolution, win=7³)
  │                       输出: (1728, 192)
  │
  ├─ PatchMerging3D     → 输出: (6³, 384) = (216, 384)
  │
  ├─ Stage 3            → 6× SwinBlock3D (384 dim, 6³ resolution, win=6³)
  │                       输出: (216, 384)
  │
  ├─ PatchMerging3D     → 输出: (3³, 768) = (27, 768)
  │
  ├─ Stage 4            → 2× SwinBlock3D (768 dim, 3³ resolution, win=3³)
  │                       输出: (27, 768)
  │
  └─ Classification Head:
       mean(token) → LayerNorm(768) → Linear(768, 2) → logits (B, 2)
```

### 默认 Stage 配置（以 `volume=(96,96,96)`, `patch=(4,4,4)` 为例）

| Stage | 输入分辨率 (D×H×W) | 通道数 | Blocks | Window Size | Heads |
|-------|-------------------|--------|--------|-------------|-------|
| 1 | 24×24×24 | 96 | 2 | 7×7×7 | 3 |
| 2 | 12×12×12 | 192 | 2 | 7×7×7 | 6 |
| 3 | 6×6×6 | 384 | 6 | 6×6×6 | 12 |
| 4 | 3×3×3 | 768 | 2 | 3×3×3 | 24 |

> 注：当分辨率小于 window_size 时，window 自动收缩到等于分辨率，且该 stage 中不再使用 shifted window。

---

## 模型参数说明

### `sMRI3DSwinTransformer` 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_classes` | `int` | `2` | 分类类别数 |
| `in_chans` | `int` | `1` | 输入通道数（灰度 MRI 为 1） |
| `patch_size` | `Tuple[int,int,int]` | `(4,4,4)` | 3D patch 大小 |
| `embed_dim` | `int` | `96` | Stage 1 嵌入维度 |
| `depths` | `Tuple[int,...]` | `(2,2,6,2)` | 每个 Stage 的 Block 数量 |
| `num_heads` | `Tuple[int,...]` | `(3,6,12,24)` | 每个 Stage 的注意力头数 |
| `window_size` | `Tuple[int,int,int]` | `(7,7,7)` | 3D 窗口大小 |
| `mlp_ratio` | `float` | `4.0` | MLP 隐藏层扩展比例 |
| `qkv_bias` | `bool` | `True` | QKV 投影是否使用 bias |
| `drop_rate` | `float` | `0.0` | MLP/投影层的 Dropout 率 |
| `attn_drop_rate` | `float` | `0.0` | 注意力权重的 Dropout 率 |
| `drop_path_rate` | `float` | `0.1` | Stochastic Depth 衰减率 |
| `input_resolution` | `Tuple[int,int,int]` | `(96,96,96)` | 输入体积空间尺寸 (D,H,W) |
| `use_checkpoint` | `bool` | `False` | 是否使用梯度检查点（省显存） |

### 模型参数量估算

| 配置 | 参数量（约） |
|------|------------|
| 默认配置 (embed_dim=96, depths=(2,2,6,2)) | ~40M |
| Tiny (embed_dim=48, depths=(2,2,2,2)) | ~10M |
| Larger (embed_dim=128, depths=(2,2,18,2)) | ~88M |

---

## 使用方法

### 快速开始

```python
import torch
from models.swin_transformer_3d import sMRI3DSwinTransformer

# 1. 初始化模型（二分类）
model = sMRI3DSwinTransformer(
    num_classes=2,
    in_chans=1,
    patch_size=(4, 4, 4),
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=(7, 7, 7),
    input_resolution=(96, 96, 96),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. 准备输入（B, C, D, H, W）
x = torch.randn(2, 1, 96, 96, 96).to(device)  # batch_size=2

# 3. 前向传播
logits = model(x)          # (2, 2)
probs = torch.softmax(logits, dim=1)
print(probs)               # tensor([[0.48, 0.52], ...])
```

### 训练循环示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.swin_transformer_3d import sMRI3DSwinTransformer

# --- 模型 ---
model = sMRI3DSwinTransformer(num_classes=2).cuda()

# --- 损失与优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

# --- 训练循环 ---
model.train()
for epoch in range(num_epochs):
    for volumes, labels in train_loader:
        volumes = volumes.cuda()   # (B, 1, D, H, W)
        labels = labels.cuda()     # (B,)

        logits = model(volumes)    # (B, num_classes)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 验证/测试

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for volumes, labels in test_loader:
        volumes = volumes.cuda()
        labels = labels.cuda()

        logits = model(volumes)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {100.0 * correct / total:.2f}%")
```

### 特征提取（迁移学习 / 下游任务）

```python
model = sMRI3DSwinTransformer(num_classes=2)
model.eval()

with torch.no_grad():
    features = model.extract_features(volumes)  # (B, 768)
    # 可用于聚类、检索、或接入其他下游模型
```

---

## 文件结构

```
sMRI-3D-SwinTransformer/models/
├── __init__.py                   # 导出 sMRI3DSwinTransformer
├── swin_transformer_3d.py        # 完整模型实现
└── MODEL_USAGE.md                # 本说明文件
```

### 模块一览

| 类/函数 | 说明 |
|---------|------|
| `Mlp` | 两层 MLP（GELU + Dropout） |
| `PatchEmbed3D` | 3D 卷积 patch 嵌入 |
| `PatchMerging3D` | 下采样（2×2×2 合并） |
| `WindowAttention3D` | 3D 窗口多头自注意力 |
| `SwinTransformerBlock3D` | Swin 块（W-MSA + SW-MSA） |
| `_window_partition_3d` | 3D 特征图 → 窗口切分 |
| `_window_reverse_3d` | 窗口 → 3D 特征图还原 |
| `BasicLayer3D` | 一个 Stage 的完整层 |
| `SwinTransformer3D` | 4 Stage Backbone |
| `sMRI3DSwinTransformer` | Backbone + 分类头 |

---

## 已知 TODO 项（待实现）

以下组件已预留接口和框架代码，标注了 `# TODO`，需要根据具体需求填充实现：

| 位置 | 内容 | 优先级 | 说明 |
|------|------|--------|------|
| `WindowAttention3D.__init__` | `relative_position_index` 构建 | 中 | 3D 相对位置偏置表的索引映射，已有参数表 `relative_position_bias_table`，缺索引计算 |
| `WindowAttention3D.forward` | 相对位置偏置加法 | 中 | `# TODO` 处已写好注释掉的代码模板，补完索引后取消注释即可 |
| `_build_3d_attention_mask` | 3D shifted window attention mask | 中 | 独立函数，需实现 cyclic shift 后的 cross-window masking |
| `SwinTransformerBlock3D.__init__` | `attn_mask` 注册 | 中 | 已写好注释掉的注册代码，mask 函数实现后取消注释 |
| `WindowAttention3D.forward` | attention mask 应用 | 中 | `# TODO` 处已写好注释掉的模板代码 |

每个 TODO 处都包含：
- 函数签名和参数说明
- 预期输入输出形状
- 实现步骤大纲（注释形式）
- 注释掉的可用代码模板

---

## 依赖

```
torch >= 1.10
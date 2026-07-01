# FC-BrainNetworkTransformer 模型使用说明

## 模型概述

经典的 BrainNetworkTransformer，将功能连接（FC）矩阵中的每个脑区（ROI）视为一个 token，通过 Transformer 自注意力机制捕获脑网络拓扑模式，实现脑疾病二分类。

### 架构流程

```
输入: (B, num_rois, num_rois)  FC 功能连接矩阵
  │
  ├─ ROIEmbedding:          每个 ROI 的连接剖面 (num_rois 维) → 线性投影 → d_model
  │                         输出: (B, num_rois, d_model)
  │
  ├─ + ROIPositionalEncoding:  每个 ROI 索引的可学习位置编码
  │
  ├─ N× TransformerEncoderLayer:
  │     ┌─ LayerNorm → MultiHeadSelfAttention → Residual
  │     └─ LayerNorm → FeedForwardNetwork (GELU) → Residual
  │                         输出: (B, num_rois, d_model)
  │
  └─ Classification Head:
       mean(token) → LayerNorm(d_model) → Linear(d_model, num_classes)
       输出: (B, num_classes)  ← 分类 logits
```

### 默认参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_rois` | 116 | 脑区数量（AAL 模板） |
| `d_model` | 256 | Token 嵌入维度 |
| `num_layers` | 6 | Transformer Encoder 层数 |
| `num_heads` | 8 | 注意力头数 |
| `ffn_dim` | 1024 (4×d_model) | FFN 隐藏层维度 |
| `dropout` | 0.1 | 全局 Dropout 率 |
| `pool_method` | `"mean"` | Token 聚合方式（当前仅支持 mean） |

---

## 模型参数说明

### `FCBrainNetworkClassifier` 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_classes` | `int` | `2` | 分类类别数 |
| `num_rois` | `int` | `116` | 脑区 ROI 数量 |
| `d_model` | `int` | `256` | Token 嵌入维度 |
| `num_layers` | `int` | `6` | Transformer 层数 |
| `num_heads` | `int` | `8` | 注意力头数 |
| `ffn_dim` | `int \| None` | `None` | FFN 隐藏维度（None = 4×d_model） |
| `dropout` | `float` | `0.1` | Dropout 率 |
| `qkv_bias` | `bool` | `True` | QKV 投影是否带 bias |
| `pool_method` | `str` | `"mean"` | Token 聚合：`"mean"` 或 `"cls"`（cls 待实现） |

### 模型参数量估算

| 配置 | 参数量（约） |
|------|------------|
| 默认 (d_model=256, layers=6, heads=8) | ~8M |
| Light (d_model=128, layers=4, heads=4) | ~2M |
| Large (d_model=512, layers=8, heads=16) | ~30M |

---

## 使用方法

### 快速开始

```python
import torch
from models.brain_network_transformer import FCBrainNetworkClassifier

# 1. 初始化模型（二分类，AAL 116 ROI）
model = FCBrainNetworkClassifier(
    num_classes=2,
    num_rois=116,
    d_model=256,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. 准备输入（B, num_rois, num_rois）
fc_matrix = torch.randn(4, 116, 116).to(device)  # batch_size=4

# 3. 前向传播
logits = model(fc_matrix)     # (4, 2)
probs = torch.softmax(logits, dim=1)
```

### 训练循环示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from models.brain_network_transformer import FCBrainNetworkClassifier

model = FCBrainNetworkClassifier(
    num_classes=2,
    num_rois=116,
    d_model=256,
    num_layers=6,
    num_heads=8,
).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

model.train()
for epoch in range(num_epochs):
    for fc_matrices, labels in train_loader:
        fc_matrices = fc_matrices.cuda()   # (B, 116, 116)
        labels = labels.cuda()             # (B,)

        logits = model(fc_matrices)        # (B, num_classes)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 验证/测试

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for fc_matrices, labels in test_loader:
        fc_matrices = fc_matrices.cuda()
        labels = labels.cuda()

        logits = model(fc_matrices)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {100.0 * correct / total:.2f}%")
```

### 特征提取

```python
model = FCBrainNetworkClassifier(num_classes=2)
model.eval()

with torch.no_grad():
    features = model.extract_features(fc_matrices)  # (B, 256)
    # 可用于可视化、聚类或接入其他分类器
```

---

## 文件结构

```
FC-BrainNetworkTransformer/models/
├── __init__.py                    # 导出 FCBrainNetworkClassifier
├── brain_network_transformer.py   # 完整模型实现
└── MODEL_USAGE.md                 # 本说明文件
```

### 模块一览

| 类/函数 | 说明 |
|---------|------|
| `MultiHeadSelfAttention` | 标准多头自注意力 |
| `FeedForwardNetwork` | 两层 MLP（GELU + Dropout） |
| `TransformerEncoderLayer` | 单个 Transformer 编码层（MSA + FFN + 残差） |
| `ROIEmbedding` | FC 剖面 → Token 嵌入（Linear + LN + Dropout） |
| `ROIPositionalEncoding` | 可学习的 ROI 位置编码 |
| `BrainNetworkTransformer` | Backbone（Embed + Pos + N×Encoder + LN） |
| `FCBrainNetworkClassifier` | 完整分类器（Backbone + Mean Pool + Head） |

---

## 已知 TODO 项（待实现）

| 位置 | 内容 | 优先级 | 说明 |
|------|------|--------|------|
| `BrainNetworkTransformer.__init__` | `[CLS]` token | 低 | 已写好注释，如需 cls pooling 则取消注释 |
| `FCBrainNetworkClassifier.forward` | `pool_method="cls"` | 低 | cls 分支抛 `NotImplementedError`，需配合上面实现 |
| `FCBrainNetworkClassifier.get_attention_maps` | 注意力图提取 | 低 | 预留接口，需修改 MSA forward 返回 attn weights |

---

## 依赖

```
torch >= 1.10
# 2026.7.24_MODEL_ver0.1 — 多模态脑疾病分类模型

基于多模态神经影像的脑疾病三分类模型，融合 **结构 MRI (sMRI)** 与 **功能 MRI (fMRI)** 特征，用于区分 **正常对照 (HC)**、**精神分裂症 (Schizophrenia，COBRE)** 和 **注意缺陷多动障碍 (ADHD，ADHD-200)**。

## 架构概览

```
                         ┌─────────────────────────────────────────────────────┐
                         │                  BrainDiseaseModel                  │
                         └─────────────────────────────────────────────────────┘

  ┌─────────────────── 结构分支 (Structural) ───────────────────┐
  │                                                              │
  │  sMRI 体素 (B,1,D,H,W) ──→ SMRIEncoder3D ──→ z_voxel ──┐   │
  │  sMRI 形态学 (B,246,6)  ──→ MorphEncoder   ──→ z_morph ─┤   │
  │                                                          │   │
  │                    StructuralGate (双门控加权融合) ←──────┘   │
  │                            │                                │
  │                      z_structure (B,64)                     │
  └────────────────────────────┼────────────────────────────────┘
                               │
  ┌─────────────────── 功能分支 (Functional) ──────────────────┐
  │                                                              │
  │  ROI BN (B,246,T_bn) ──→ LearnableFC ──→ FCEncoder2D ──→ z_bn ──┐  │
  │  ROI AAL (B,246,T_aal) ──→ LearnableFC ──→ FCEncoder2D ──→ z_aal ─┤  │
  │                                                                      │  │
  │                       FunctionalAttention (softmax 加权 + 残差) ←───┘  │
  │                                   │                                   │
  │                            z_functional (B,64)                        │
  └───────────────────────────────────┼───────────────────────────────────┘
                                      │
                    DiseaseFusion (Cross-Attention + 模态嵌入)
                                      │
                                 z_disease (B,64)
                                      │
                              Classifier (Linear → ReLU → Dropout → Linear)
                                      │
                                logits (B, 3)
```

### 关键模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| `SMRIEncoder3D` | `encoders.py` | 3D CNN 编码 sMRI 体素，输出隐向量 |
| `MorphEncoder` | `encoders.py` | ROI 级别形态学特征 (6 维) → MLP + MultiheadAttention 池化 |
| `FCEncoder2D` | `encoders.py` | 2D CNN 编码功能连接矩阵 |
| `LearnableFC` | `fusion.py` | 从 ROI 时间序列学习功能连接矩阵 (Q·K^T → softmax → 对称化) |
| `StructuralGate` | `fusion.py` | 双门控融合：两个独立 Sigmoid 门分别加权体素和形态特征 |
| `FunctionalAttention` | `fusion.py` | 图谱自适应融合：softmax 学习 BN/AAL 两个图谱的最优加权 |
| `DiseaseFusion` | `fusion.py` | Disease Token Cross-Attention + 模态身份嵌入，融合结构/功能特征 |
| `SupConLoss` | `losses.py` | 监督对比损失 (Khosla et al., NeurIPS 2020)，同类靠近、异类远离 |

## 文件结构

```
2026.7.24_MODEL_ver0.1/
├── __init__.py        # 包导出
├── augmentation.py    # 数据增强 (SMRIAugment, FCAugment)
├── encoders.py        # 编码器 (SMRIEncoder3D, FCEncoder2D, MorphEncoder)
├── fusion.py          # 融合模块 (StructuralGate, FunctionalAttention, LearnableFC, DiseaseFusion)
├── losses.py          # 损失函数 (SupConLoss)
├── model.py           # 主模型 BrainDiseaseModel
├── trainer.py         # 训练/验证循环 + 指标计算
└── main.py            # 训练入口 (argparse, 数据加载, 训练循环)
```

## 数据集

| 数据集 | 疾病 | 来源 |
|--------|------|------|
| COBRE | 精神分裂症 (Schizophrenia) | http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html |
| ADHD-200 | 注意缺陷多动障碍 (ADHD) | http://fcon_1000.projects.nitrc.org/indi/adhd200/ |

### 输入模态

每个样本包含 4 种输入：

1. **sMRI 体素** `(1, D, H, W)` — T1 加权结构像
2. **sMRI 形态学** `(246, 6)` — Brainnetome 图谱 246 个脑区的形态学特征
3. **ROI BN** `(246, T_bn)` — Brainnetome 图谱的 BOLD 时间序列
4. **ROI AAL** `(246, T_aal)` — AAL 图谱的 BOLD 时间序列

标签映射：`0 = HC`, `1 = Schizophrenia`, `2 = ADHD`

## 快速开始

### 环境依赖

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy

### 训练

```bash
# 默认参数
python -m 2026.7.24_MODEL_ver0.1.main

# 自定义参数
python -m 2026.7.24_MODEL_ver0.1.main \
    --epochs 150 \
    --batch_size 16 \
    --lr 0.0005 \
    --latent_dim 128 \
    --dropout 0.4 \
    --lambda_contrast 0.05 \
    --seed 42 \
    --cobre_dir /path/to/COBRE \
    --adhd_dir /path/to/ADHD-200 \
    --save_path ./checkpoints/best_model.pth
```

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 |
| `--lr` | 0.001 | 学习率 (AdamW) |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--latent_dim` | 64 | 隐空间维度 |
| `--dropout` | 0.3 | Dropout 比例 |
| `--num_classes` | 3 | 分类类别数 |
| `--label_smoothing` | 0.02 | 标签平滑 |
| `--lambda_contrast` | 0.1 | 监督对比损失权重（设 0 禁用） |
| `--early_stop` | 15 | 早停 patience |
| `--seed` | 42 | 随机种子 |

### 训练策略

- **损失函数**: CrossEntropyLoss (label smoothing=0.02) + SupConLoss (λ=0.1)
- **优化器**: AdamW
- **调度器**: CosineAnnealingLR (eta_min = lr×0.01)
- **梯度裁剪**: max_norm = 1.0
- **早停**: 15 epochs 无提升即停止
- **数据分割**: 按类别分层 70% / 15% / 15% (训练/验证/测试)
- **数据增强**: sMRI 高斯噪声 (σ=0.005, p=0.5)，ROI TS 噪声 + 随机遮蔽 (p=0.3, drop_rate=0.02)

## 训练输出

训练过程自动保存：
- `best_model_epoch{N}_train{loss}_val{loss}_acc{acc}.pth` — 最佳模型权重
- `best_model_history.json` — 训练历史 (loss / accuracy / per_class metrics)

## 引用

Khosla, P., et al. "Supervised Contrastive Learning." *NeurIPS*, 2020.

---

*Version 0.1 — 2026.7.24*

# 2026.7.31_MODEL_ver0.2

**Multimodal Brain Disease Classification v0.2** — An upgraded deep learning model for classifying **Healthy Control (HC)**, **Schizophrenia (COBRE)**, and **ADHD (ADHD-200)** via structural & functional MRI fusion.

> **⚠️  v0.1 → v0.2 主要更新见下方 Changelog 章节。**

---

## Changelog (v0.1 → v0.2)

| Category | v0.1 | v0.2 |
|----------|------|------|
| **Functional Encoder** | LearnableFC + FCEncoder2D (2D CNN) | **GATEncoder** (2-layer Graph Attention, direct ROI TS → message passing) |
| **Brain Atlases** | BN + AAL (dual-atlas, fused via FunctionalAttention) | **BN only** (AAL removed; simpler pipeline, single functional branch) |
| **MorphEncoder** | MHA + Query Token pooling (~12k params) | **Scalar attention pooling** (~640 params, better for small-sample regime) |
| **StructuralGate** | Pure dual-gate | **+ Cosine-similarity trust gate** (conflict → fallback to mean) |
| **DiseaseFusion** | 1 disease token + Cross-Attention over 2 KV | **3 disease tokens** (one per class), per-token projection, sigmoid modality mixing, token_bias |
| **DiseasePrototypeHead** | — | **NEW**: per-class prototype → token-prototype L2 loss + token diversity loss |
| **Domain Adaptation** | — | **NEW**: CDAN domain discriminator, HC-only discriminator, ComBat site correction wrappers |
| **Loss Functions** | CE + SupCon (2 total) | CE + SupCon + Ortho + Proto + Domain + MMD + HC-Domain + Token Diversity (8 total, switchable) |
| **Sampling** | Random shuffle | **BalancedBatchSampler** (fixed N per class, minority oversampling) |
| **Normalization** | — | Instance norm (per-sample z-score on sMRI/ROI), per-ROI z-score on BOLD |
| **Training Monitor** | Loss + Acc + Per-class | **+ Silhouette Score, DB Index, group distances, prototype cosine matrix, gradient ratios** |
| **Batch Size** | 8 | 24 |
| **Output** | JSON history | JSON + CSV (classification + generalization) |
| **LearnableFC** | softmax(QK^T/√d) | **max-abs normalization** to [-1,1] (preserves anti-correlation) |

### Summary of Design Philosophy Shifts

1. **From 2D CNN to Graph Attention** — ROI time series are a graph (246 nodes × T timepoints); GAT with learnable attention naturally captures inter-ROI relationships without an intermediate FC matrix bottleneck.
2. **Single Atlas Simplification** — AAL was removed after experiments showed negligible gain over BN-only, reducing model complexity and overfitting risk.
3. **Multi-Token Disease Representation** — One token per class disentangles HC/SZ/ADHD representations; each token learns a disease-specific viewpoint.
4. **Domain-Aware Training** — COBRE and ADHD-200 are different sites/datasets. ComBat correction + CDAN domain adversarial + HC-only domain alignment mitigate site confounds.
5. **Orthogonality & Prototype Regularization** — Orthogonality pushes structural and functional branches to learn complementary (not redundant) features. Prototype loss anchors each class to a learnable centroid for better separation.

---

## Architecture Overview

```
                         ┌─────────────────────────────────────────────────────┐
                         │                  BrainDiseaseModel                  │
                         └─────────────────────────────────────────────────────┘

  ┌─────────────────── Structural Branch ──────────────────────┐
  │                                                              │
  │  sMRI voxels (B,1,D,H,W) ──→ SMRIEncoder3D ──→ z_voxel ──┐ │
  │  sMRI morph    (B,246,6)  ──→ MorphEncoder   ──→ z_morph ─┤ │
  │                                                            │ │
  │         StructuralGate (dual-gate + cosine trust) ←────────┘ │
  │                            │                                  │
  │                      z_structure (B,64)                       │
  └────────────────────────────┼──────────────────────────────────┘
                               │
  ┌─────────────────── Functional Branch ─────────────────────┐
  │                                                              │
  │  ROI BN (B,246,T_bn)  ──→ GATEncoder (2-layer GAT) ──→ z_functional (B,64)  │
  │                                                              │
  └────────────────────────────┼─────────────────────────────────┘
                               │
                    DiseaseFusion (3 disease tokens, per-token mixing)
                               │
                     disease_tokens (B, 3, 64) ──→ DiseasePrototypeHead
                               │                          │
                         z_disease (B, 192)          proto_logits (B, 3)
                               │
                      Classifier (MLP → 3 classes)
                               │
                          logits (B, 3)
```

## Key Modules

| Module | File | Description |
|--------|------|-------------|
| `SMRIEncoder3D` | `encoders.py` | 3D CNN encoder for sMRI voxel data |
| `MorphEncoder` | `encoders.py` | Scalar attention pooling over 246 ROI × 6 morphological features |
| `GATEncoder` | `encoders.py` | 2-layer Graph Attention Network — ROI TS directly → graph message passing |
| `StructuralGate` | `encoders.py` | Dual-gate fusion with cosine-similarity disagreement fallback |
| `DiseaseFusion` | `encoders.py` | 3 per-class disease tokens with sigmoid modality mixing + token_bias |
| `BrainDiseaseModel` | `encoders.py` | Top-level model combining all encoders + fusion + classifier |
| `DiseasePrototypeHead` | `proto_head.py` | Per-class learnable prototypes → token-prototype diagonal scores |
| `GradientReversalLayer` | `domain_modules.py` | GRL for domain adversarial training |
| `DomainDiscriminator` | `domain_modules.py` | CDAN conditional domain discriminator (COBRE vs ADHD-200) |
| `HCDomainDiscriminator` | `domain_modules.py` | HC-only linear discriminator — aligns healthy controls across sites |
| `DomainWrapper` | `domain_modules.py` | Adds domain/health labels to dataset items |
| `CombatMorphWrapper` | `domain_modules.py` | Replaces morph features with ComBat site-corrected version |
| `CombatFcWrapper` | `domain_modules.py` | Replaces ROI TS with ComBat-corrected FC matrix |
| `SupConLoss` | `losses.py` | Supervised Contrastive Loss |
| `DomainAwareSupConLoss` | `losses.py` | Cross-domain contrastive: same-disease-same-site=1.0, same-disease-cross-site=0.4 |
| `BalancedBatchSampler` | `sampler.py` | Fixed N samples per class per batch, minority oversampling |
| `train_one_epoch` / `validate` | `train_utils.py` | Training loop with 8 switchable losses + gradient ratio monitoring |

## Loss Functions

| Loss | Flag | Default λ | Purpose |
|------|------|-----------|---------|
| CrossEntropy | (always) | 1.0 | Primary classification, label smoothing 0.02 |
| SupConLoss | `--lambda_contrast` | 0.1 | Same-class attraction, different-class repulsion |
| Orthogonality | `--lambda_ortho` | 0.05 | Push z_structure ⟂ z_functional for complementary features |
| Prototype | `--lambda_proto` | 0.5 | L2 distance: disease token → class prototype |
| Token Diversity | (nested in proto) | 0.3×λ_proto | Encourage 3 tokens to encode complementary info |
| Domain Adversarial | `--lambda_domain` | 0.0 (experimental) | CDAN: fool domain classifier from latent features |
| HC Domain | `--lambda_hc_domain` | 0.1 | HC-only: align healthy controls across COBRE/ADHD-200 |
| MMD | `--lambda_mmd` | 0.0 (experimental) | Per-class MMD: align feature distributions across sites |

## File Structure

```
2026.7.31_MODEL_ver0.2/
├── domain_modules.py    # Domain wrappers, GRL, discriminators, ComBat wrappers
├── encoders.py          # All encoders + fusion + BrainDiseaseModel (merged from v0.1 fusion.py/model.py)
├── losses.py            # SupConLoss, DomainAwareSupConLoss, orthogonality, MMD, prototype losses
├── main.py              # Training entry point
├── proto_head.py        # DiseasePrototypeHead
├── sampler.py           # BalancedBatchSampler
└── train_utils.py       # Training/validation loop, metrics, group distance, Silhouette, DB Index
```

## Datasets

| Dataset | Condition | Source |
|---------|-----------|--------|
| COBRE | Schizophrenia | http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html |
| ADHD-200 | ADHD | http://fcon_1000.projects.nitrc.org/indi/adhd200/ |

### Input Modalities (per sample)

| Modality | Shape | Description |
|----------|-------|-------------|
| sMRI voxels | `(1, D, H, W)` | T1-weighted structural image |
| sMRI morphology | `(246, 6)` | Morphological features of 246 Brainnetome ROIs |
| ROI BN | `(246, T_bn)` | BOLD time series — Brainnetome atlas |

Label mapping: `0 = HC`, `1 = Schizophrenia`, `2 = ADHD`

## Quick Start

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy, scikit-learn

### Training

```bash
# Default parameters
python -m 2026.7.31_MODEL_ver0.2.main

# With ComBat site correction (recommended)
python -m 2026.7.31_MODEL_ver0.2.main --combat_morph --combat_fc

# Full custom run
python -m 2026.7.31_MODEL_ver0.2.main \
    --epochs 150 \
    --batch_size 24 \
    --lr 0.001 \
    --latent_dim 64 \
    --dropout 0.3 \
    --lambda_contrast 0.1 \
    --lambda_ortho 0.05 \
    --lambda_proto 0.5 \
    --lambda_hc_domain 0.1 \
    --lambda_domain 0.0 \
    --lambda_mmd 0.0 \
    --combat_morph \
    --combat_fc \
    --seed 42 \
    --save_path ./checkpoints/best_model.pth
```

### Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 24 | Batch size (× 3 classes = 8/class via BalancedBatchSampler) |
| `--lr` | 0.001 | Learning rate (AdamW) |
| `--weight_decay` | 0.01 | Weight decay |
| `--latent_dim` | 64 | Latent space dimension |
| `--dropout` | 0.3 | Dropout rate |
| `--num_classes` | 3 | Number of classes |
| `--label_smoothing` | 0.02 | Label smoothing |
| `--lambda_contrast` | 0.1 | SupCon loss weight (set 0 to disable) |
| `--lambda_ortho` | 0.05 | Orthogonality constraint (0.01~0.1 recommended) |
| `--lambda_proto` | 0.5 | Prototype loss weight |
| `--lambda_hc_domain` | 0.1 | HC-only domain adversarial weight |
| `--lambda_domain` | 0.0 | CDAN domain adversarial (experimental) |
| `--lambda_mmd` | 0.0 | MMD domain alignment (experimental) |
| `--early_stop` | 15 | Early stopping patience |
| `--seed` | 42 | Random seed |
| `--combat_morph` | False | Use ComBat-corrected morph features |
| `--combat_fc` | False | Use ComBat-corrected FC (replaces ROI TS) |

## Outputs

Training automatically saves under `OUTPUT/<timestamp>/`:

- `best_model_epoch{N}_*.pth` — Best model weights
- `best_model_history.json` — Training history
- `分类性能.csv` — Per-epoch classification metrics (accuracy, macro-F1, weighted-F1, AUC, per-class recall/F1)
- `泛化能力.csv` — Generalization gap tracking (train/val accuracy gap, loss gap, balanced accuracy)

---

---

# 2026.7.31_MODEL_ver0.2

**多模态脑疾病分类模型 v0.2** — 升级版深度学习模型，融合结构与功能 MRI 特征，区分 **正常对照 (HC)**、**精神分裂症 (COBRE)** 和 **ADHD (ADHD-200)**。

> **⚠️  v0.1 → v0.2 主要更新见下方更新日志章节。**

---

## 更新日志 (v0.1 → v0.2)

| 类别 | v0.1 | v0.2 |
|------|------|------|
| **功能编码器** | LearnableFC + FCEncoder2D (2D CNN) | **GATEncoder**（2 层图注意力网络，ROI 时间序列直接做图消息传递） |
| **脑图谱** | BN + AAL（双图谱，FunctionalAttention 融合） | **仅 BN**（移除 AAL，简化流水线，单功能分支） |
| **MorphEncoder** | MHA + Query Token 池化（~12k 参数） | **标量注意力池化**（~640 参数，适配小样本） |
| **StructuralGate** | 纯双门控 | **+ 余弦相似度信任机制**（特征冲突时退化为平均） |
| **DiseaseFusion** | 1 个 disease token + Cross-Attention | **3 个 disease token**（每类一个），独立投影，sigmoid 模态混合，token_bias |
| **DiseasePrototypeHead** | 无 | **新增**：每类原型 → token-prototype L2 损失 + token 多样性损失 |
| **域适应** | 无 | **新增**：CDAN 域判别器、HC 域判别器、ComBat 站点校正包装器 |
| **损失函数** | CE + SupCon（共 2 个） | CE + SupCon + Ortho + Proto + Domain + MMD + HC-Domain + Token Div（共 8 个，可开关） |
| **采样策略** | 随机 shuffle | **BalancedBatchSampler**（固定每类样本数，少数类过采样） |
| **归一化** | 无 | Instance norm（sMRI/ROI 逐样本 z-score），Per-ROI z-score |
| **训练监控** | Loss + Acc + Per-class | **+ Silhouette Score, DB Index, 组距离, prototype 余弦矩阵, 梯度比例** |
| **批次大小** | 8 | 24 |
| **输出** | JSON 历史 | JSON + CSV（分类性能 + 泛化能力） |
| **LearnableFC** | softmax(QK^T/√d) | **max-abs 归一化** 至 [-1,1]（保留反相关的脑区） |

### 设计理念变化总结

1. **从 2D CNN 到图注意力** — ROI 时间序列本质是图（246 节点 × T 时间点）；GAT 自学习注意力直接捕获脑区间关系，跳过 FC 矩阵中间瓶颈。
2. **单图谱简化** — 实验发现 AAL 对 BN 增量收益可忽略，移除后降低模型复杂度和过拟合风险。
3. **多 Token 疾病表征** — 每类一个 token 解耦 HC/SZ/ADHD 的表征空间；每个 token 学习疾病专属的视角。
4. **域感知训练** — COBRE 和 ADHD-200 来自不同站点。ComBat 校正 + CDAN 域对抗 + HC 域对齐缓解站点混杂。
5. **正交性与原型正则化** — 正交性推动结构/功能分支学习互补（非冗余）特征。原型损失将每类锚定到可学习中心，增强类间分离。

---

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
  │         StructuralGate (双门控 + 余弦信任机制) ←──────────┘   │
  │                            │                                │
  │                      z_structure (B,64)                     │
  └────────────────────────────┼────────────────────────────────┘
                               │
  ┌─────────────────── 功能分支 (Functional) ─────────────────┐
  │                                                              │
  │  ROI BN (B,246,T_bn)  ──→ GATEncoder (2 层 GAT) ──→ z_functional (B,64)  │
  │                                                              │
  └────────────────────────────┼─────────────────────────────────┘
                               │
                    DiseaseFusion (3 个 disease token，独立模态混合)
                               │
                     disease_tokens (B, 3, 64) ──→ DiseasePrototypeHead
                               │                          │
                         z_disease (B, 192)          proto_logits (B, 3)
                               │
                      Classifier (MLP → 3 分类)
                               │
                          logits (B, 3)
```

## 关键模块说明

| 模块 | 文件 | 功能说明 |
|------|------|----------|
| `SMRIEncoder3D` | `encoders.py` | 3D CNN 编码 sMRI 体素数据 |
| `MorphEncoder` | `encoders.py` | 标量注意力池化处理 246 ROI × 6 形态学特征 |
| `GATEncoder` | `encoders.py` | 2 层图注意力网络 — ROI TS 直接做图消息传递 |
| `StructuralGate` | `encoders.py` | 双门控融合 + 余弦相似度冲突回退 |
| `DiseaseFusion` | `encoders.py` | 3 个疾病 token + sigmoid 模态混合 + token_bias |
| `BrainDiseaseModel` | `encoders.py` | 顶层模型，组合所有编码器 + 融合 + 分类器 |
| `DiseasePrototypeHead` | `proto_head.py` | 可学习类别原型 → token-prototype 对角得分 |
| `GradientReversalLayer` | `domain_modules.py` | GRL 梯度反转层，用于域对抗训练 |
| `DomainDiscriminator` | `domain_modules.py` | CDAN 条件域判别器（COBRE vs ADHD-200） |
| `HCDomainDiscriminator` | `domain_modules.py` | HC-only 线性判别器 — 对齐跨站点健康对照 |
| `DomainWrapper` | `domain_modules.py` | 为数据集样本添加域标签和健康标签 |
| `CombatMorphWrapper` | `domain_modules.py` | 替换为 ComBat 站点校正后的形态学特征 |
| `CombatFcWrapper` | `domain_modules.py` | 替换为 ComBat 校正后的 FC 矩阵 |
| `SupConLoss` | `losses.py` | 监督对比损失 |
| `DomainAwareSupConLoss` | `losses.py` | 跨域对比损失：同病同域=1.0，同病跨域=0.4 |
| `BalancedBatchSampler` | `sampler.py` | 每 batch 固定各类样本数，少数类过采样 |
| `train_one_epoch` / `validate` | `train_utils.py` | 训练循环，含 8 个可开关损失 + 梯度比例监控 |

## 损失函数

| 损失 | 开关 | 默认 λ | 作用 |
|------|------|--------|------|
| CrossEntropy | 始终 | 1.0 | 主分类损失，label smoothing 0.02 |
| SupConLoss | `--lambda_contrast` | 0.1 | 同类靠近，异类远离 |
| Orthogonality | `--lambda_ortho` | 0.05 | 推动 z_structure ⟂ z_functional，学习互补特征 |
| Prototype | `--lambda_proto` | 0.5 | L2 距离：disease token → 类别原型 |
| Token Diversity | (嵌套于 proto) | 0.3×λ_proto | 鼓励 3 个 token 编码互补信息 |
| Domain Adversarial | `--lambda_domain` | 0.0（实验性） | CDAN：从隐空间特征迷惑域分类器 |
| HC Domain | `--lambda_hc_domain` | 0.1 | HC-only：对齐 COBRE/ADHD-200 中的健康对照 |
| MMD | `--lambda_mmd` | 0.0（实验性） | Per-class MMD：对齐跨域特征分布 |

## 文件结构

```
2026.7.31_MODEL_ver0.2/
├── domain_modules.py    # 域包装器、GRL、判别器、ComBat 包装器
├── encoders.py          # 所有编码器 + 融合模块 + BrainDiseaseModel（合并自 v0.1 的 fusion.py/model.py）
├── losses.py            # SupConLoss, DomainAwareSupConLoss, 正交性, MMD, prototype 损失
├── main.py              # 训练入口
├── proto_head.py        # DiseasePrototypeHead
├── sampler.py           # BalancedBatchSampler
└── train_utils.py       # 训练/验证循环, 指标, 组距离, Silhouette, DB Index
```

## 数据集

| 数据集 | 疾病 | 来源 |
|--------|------|------|
| COBRE | 精神分裂症 | http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html |
| ADHD-200 | 注意缺陷多动障碍 | http://fcon_1000.projects.nitrc.org/indi/adhd200/ |

### 输入模态（每个样本）

| 模态 | 形状 | 说明 |
|------|------|------|
| sMRI 体素 | `(1, D, H, W)` | T1 加权结构像 |
| sMRI 形态学 | `(246, 6)` | Brainnetome 图谱 246 个脑区的形态学特征 |
| ROI BN | `(246, T_bn)` | Brainnetome 图谱的 BOLD 时间序列 |

标签映射：`0 = HC`, `1 = Schizophrenia`, `2 = ADHD`

## 快速开始

### 环境依赖

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy, scikit-learn

### 训练

```bash
# 默认参数
python -m 2026.7.31_MODEL_ver0.2.main

# 使用 ComBat 站点校正（推荐）
python -m 2026.7.31_MODEL_ver0.2.main --combat_morph --combat_fc

# 完整自定义训练
python -m 2026.7.31_MODEL_ver0.2.main \
    --epochs 150 \
    --batch_size 24 \
    --lr 0.001 \
    --latent_dim 64 \
    --dropout 0.3 \
    --lambda_contrast 0.1 \
    --lambda_ortho 0.05 \
    --lambda_proto 0.5 \
    --lambda_hc_domain 0.1 \
    --lambda_domain 0.0 \
    --lambda_mmd 0.0 \
    --combat_morph \
    --combat_fc \
    --seed 42 \
    --save_path ./checkpoints/best_model.pth
```

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 24 | 批次大小（÷3 类 = 每类 8，通过 BalancedBatchSampler） |
| `--lr` | 0.001 | 学习率 (AdamW) |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--latent_dim` | 64 | 隐空间维度 |
| `--dropout` | 0.3 | Dropout 比例 |
| `--num_classes` | 3 | 分类类别数 |
| `--label_smoothing` | 0.02 | 标签平滑 |
| `--lambda_contrast` | 0.1 | SupCon 损失权重 |
| `--lambda_ortho` | 0.05 | 正交性约束（推荐 0.01~0.1） |
| `--lambda_proto` | 0.5 | 原型损失权重 |
| `--lambda_hc_domain` | 0.1 | HC-only 域对抗权重 |
| `--lambda_domain` | 0.0 | CDAN 域对抗（实验性） |
| `--lambda_mmd` | 0.0 | MMD 域对齐（实验性） |
| `--early_stop` | 15 | 早停 patience |
| `--seed` | 42 | 随机种子 |
| `--combat_morph` | False | 使用 ComBat 校正后的 morph 特征 |
| `--combat_fc` | False | 使用 ComBat 校正后的 FC（替换 ROI TS） |

## 训练输出

训练过程在 `OUTPUT/<timestamp>/` 下自动保存：

- `best_model_epoch{N}_*.pth` — 最佳模型权重
- `best_model_history.json` — 训练历史
- `分类性能.csv` — 每 epoch 分类指标（accuracy, macro-F1, weighted-F1, AUC, 各类 recall/F1）
- `泛化能力.csv` — 泛化间隙追踪（train/val accuracy gap, loss gap, balanced accuracy）

---

*Version 0.2 — 2026.7.31*

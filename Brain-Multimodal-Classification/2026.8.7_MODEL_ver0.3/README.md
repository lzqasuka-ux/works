# 2026.8.7_MODEL_ver0.3

**Multimodal Brain Disease Classification v0.3** — An upgraded deep learning model for classifying **Healthy Control (HC)**, **Schizophrenia (COBRE)**, and **ADHD (ADHD-200)** via structural & functional MRI fusion.

> **⚠️  v0.2 → v0.3 主要更新见下方 Changelog 章节。**

---

## Changelog (v0.2 → v0.3)

| Category | v0.2 | v0.3 |
|----------|------|------|
| **Functional Encoder** | Linear projection → GAT | **TemporalEncoder (1D CNN) → GAT**: temporal CNN extracts per-ROI time-dynamics (kernel 7/5 + AdaptiveAvgPool1d), shared weights across all 246 ROIs |
| **ROI Embedding** | Added after projection | **Added to raw input before temporal encoding** (breaks 246-node convergence earlier) |
| **Time Series Length** | Zero-padding to global max T | **Linear interpolation to fixed `--t_target 150`** (uniform resampling; per-ROI z-score on original length, then interpolate) |
| **ROI Preprocessing** | per-ROI z-score inside trainer | **Moved into `collate_fn`** (z-score → interpolate once, cleaner separation) |
| **Calibration Metrics** | — | **NEW**: ECE (Expected Calibration Error, 10 bins), mean/correct/wrong confidence, confidence gap |
| **Latent Space Analysis** | Silhouette + DB Index (printed) | **+ Returned structured metrics**: silhouette (cosine metric), inter/intra distance, separation ratio |
| **Multimodal Validity Probe** | — | **NEW**: LogisticRegression + StratifiedKFold probes on sMRI / FC / fusion features → `fusion_gain = fusion_acc − max(smri_acc, fc_acc)` |
| **CSV Output** | 分类性能 + 泛化能力 (2) | **+ 潜在空间可分性 + 多模态有效性 + 预测可依赖性 (5 total)** |
| **Result Visualizations** | — | **NEW: `test_img/`** — 7 figures (ROC, t-SNE, Confusion Matrix, Best round, Early vs Late, Classification performance, Training overview) |
| **File Structure** | 7 modules (encoders/fusion/proto/sampler split) | **Consolidated to 5 modules**: `models.py` (all encoders+fusion+head+augment), `trainer.py` (train+metrics+sampler), `domain.py`, `losses.py`, `main.py` |

### Design Philosophy Shifts

1. **Temporal CNN before Graph** — A 1D CNN first learns per-ROI temporal dynamics, then GAT models inter-ROI relationships. Decouples "what each ROI's signal looks like" from "how ROIs interact".
2. **Fixed-length interpolation** — Resampling all ROI series to T=150 (vs padding) removes length variability while preserving signal shape; enables batch CNN processing.
3. **Reliability-aware evaluation** — ECE + confidence metrics quantify how trustworthy predictions are, not just accuracy.
4. **Multimodal validity probing** — Logistic probes isolate each modality's contribution and compute the fusion gain, proving the fusion is actually better than the best single modality.
5. **Code consolidation** — Merging related modules reduces import complexity without changing architecture.

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
  │  ROI BN (B,246,150) ──→ TemporalEncoder (1D CNN) ──→ GAT (2-layer) ──→ z_functional (B,64)  │
  │                           (per-ROI time dynamics)      (inter-ROI message passing)         │
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
| `SMRIEncoder3D` | `models.py` | 3D CNN encoder for sMRI voxel data |
| `MorphEncoder` | `models.py` | Scalar attention pooling over 246 ROI × 6 morphological features |
| `TemporalEncoder` | `models.py` | **NEW**: shared-weight 1D CNN (kernel 7/5) extracting per-ROI temporal dynamics |
| `GATEncoder` | `models.py` | 2-layer GAT — TemporalEncoder output → graph message passing |
| `StructuralGate` | `models.py` | Dual-gate fusion with cosine-similarity disagreement fallback |
| `DiseaseFusion` | `models.py` | 3 per-class disease tokens with sigmoid modality mixing + token_bias |
| `BrainDiseaseModel` | `models.py` | Top-level model |
| `DiseasePrototypeHead` | `models.py` | Per-class learnable prototypes (merged into models.py) |
| `SMRIAugment` | `models.py` | sMRI Gaussian noise augmentation |
| `GradientReversalLayer` | `domain.py` | GRL for domain adversarial training |
| `DomainDiscriminator` | `domain.py` | CDAN conditional domain discriminator |
| `HCDomainDiscriminator` | `domain.py` | HC-only linear discriminator |
| `DomainWrapper` | `domain.py` | Adds domain/health labels to dataset items |
| `CombatMorphWrapper` | `domain.py` | ComBat site-corrected morph features |
| `CombatFcWrapper` | `domain.py` | ComBat-corrected FC matrix (replaces ROI TS) |
| `SupConLoss` | `losses.py` | Supervised Contrastive Loss |
| `orthogonality_loss` | `losses.py` | z_structure ⟂ z_functional constraint |
| `mmd_loss` / `consistency_loss` | `losses.py` | Domain alignment losses |
| `prototype_loss` / `token_diversity_loss` | `losses.py` | Prototype anchoring + token decorrelation |
| `train_one_epoch` / `validate` | `trainer.py` | Training loop + metrics + ECE calibration |
| `compute_group_distances` | `trainer.py` | Latent separability + multimodal validity probes |
| `BalancedBatchSampler` | `trainer.py` | Fixed N per class per batch, minority oversampling |

## Evaluation Metrics (NEW in v0.3)

### Classification (per epoch, CSV)

accuracy, macro-F1, weighted-F1, macro-AUC, precision/recall (macro + per-class)

### Prediction Reliability (`预测可依赖性.csv`)

| Metric | Meaning |
|--------|---------|
| ECE | Expected Calibration Error over 10 bins (↓ better) |
| mean_confidence | Average max-softmax confidence |
| correct / wrong confidence | Avg confidence on correct vs wrong predictions |
| confidence_gap | correct_conf − wrong_conf (↑ better) |

### Latent Separability (`潜在空间可分性.csv`)

| Metric | Meaning |
|--------|---------|
| latent_silhouette | Silhouette on z_disease (cosine metric, ↑ better) |
| latent_inter_distance | Mean inter-class center distance (1−cos, ↑ better) |
| latent_intra_distance | Mean intra-class sample-center distance (↓ better) |
| latent_separation_ratio | inter / intra (↑ better) |

### Multimodal Validity (`多模态有效性.csv`)

LogisticRegression probes (StratifiedKFold, 3 folds) on each latent space:

| Metric | Meaning |
|--------|---------|
| smri_acc | Probe accuracy on z_structure only |
| fc_acc | Probe accuracy on z_functional only |
| fusion_acc | Probe accuracy on z_disease (fused) |
| fusion_gain | fusion_acc − max(smri_acc, fc_acc) — **proof of fusion benefit** |

## Result Visualizations (`test_img/`)

```
test_img/
├── Overview of Model Training.png    # 训练过程总览
├── Best round score.png              # 最佳轮次得分
├── Classification performance.png    # 分类性能
├── Confusion Matrix.png              # 混淆矩阵
├── ROC.png                           # ROC 曲线
├── t-SNE.png                         # 潜在空间 t-SNE 可视化
└── Early vs Late.png                 # 早融合 vs 晚融合对比
```

## File Structure

```
2026.8.7_MODEL_ver0.3/
├── domain.py        # GRL, discriminators, dataset/ComBat wrappers (was domain_modules.py)
├── losses.py        # SupCon, orthogonality, MMD, consistency, prototype, token diversity
├── main.py          # Training entry point
├── models.py        # ALL encoders + fusion + BrainDiseaseModel + DiseasePrototypeHead + SMRIAugment
├── test_img/        # Test result visualizations (7 figures)
└── trainer.py       # Training/validation, metrics, ECE, separability probes, BalancedBatchSampler
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
| ROI BN | `(246, 150)` | BOLD time series (Brainnetome), interpolated to T=150 |

Label mapping: `0 = HC`, `1 = Schizophrenia`, `2 = ADHD`

## Quick Start

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy, scikit-learn

### Training

```bash
# Default parameters
python -m 2026.8.7_MODEL_ver0.3.main

# With ComBat site correction (recommended)
python -m 2026.8.7_MODEL_ver0.3.main --combat_morph --combat_fc

# Full custom run
python -m 2026.8.7_MODEL_ver0.3.main \
    --epochs 150 \
    --batch_size 24 \
    --t_target 150 \
    --lr 0.001 \
    --latent_dim 64 \
    --dropout 0.3 \
    --lambda_contrast 0.1 \
    --lambda_ortho 0.05 \
    --lambda_proto 0.5 \
    --lambda_hc_domain 0.1 \
    --combat_morph \
    --combat_fc \
    --seed 42
```

### Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 24 | Batch size (÷ 3 classes = 8/class) |
| `--t_target` | 150 | **NEW**: ROI time series interpolation target length |
| `--lr` | 0.001 | Learning rate (AdamW) |
| `--weight_decay` | 0.01 | Weight decay |
| `--latent_dim` | 64 | Latent space dimension |
| `--dropout` | 0.3 | Dropout rate |
| `--num_classes` | 3 | Number of classes |
| `--label_smoothing` | 0.02 | Label smoothing |
| `--lambda_contrast` | 0.1 | SupCon loss weight |
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
- `best_prototypes.pt` — Learned disease prototypes
- `history.json` — Training history
- `分类性能.csv` — Classification metrics
- `泛化能力.csv` — Generalization gap tracking
- `潜在空间可分性.csv` — Latent separability metrics
- `多模态有效性.csv` — Multimodal validity probes + fusion gain
- `预测可依赖性.csv` — ECE + confidence calibration metrics

---

---

# 2026.8.7_MODEL_ver0.3

**多模态脑疾病分类模型 v0.3** — 升级版深度学习模型，融合结构与功能 MRI 特征，区分 **正常对照 (HC)**、**精神分裂症 (COBRE)** 和 **ADHD (ADHD-200)**。

> **⚠️  v0.2 → v0.3 主要更新见下方更新日志章节。**

---

## 更新日志 (v0.2 → v0.3)

| 类别 | v0.2 | v0.3 |
|------|------|------|
| **功能编码器** | Linear 投影 → GAT | **TemporalEncoder (1D CNN) → GAT**：时间卷积提取每个 ROI 的时间动力学模式（kernel 7/5 + AdaptiveAvgPool1d），246 个 ROI 共享权重 |
| **ROI Embedding** | 投影后添加 | **提前加到原始输入**（时间编码之前），更早打破 246 节点趋同 |
| **时间序列长度** | padding 到全局 max T | **线性插值到固定 `--t_target 150`**（先按原长度 per-ROI z-score，再统一重采样） |
| **ROI 预处理** | trainer 内 per-ROI z-score | **移入 `collate_fn`**（z-score → 插值一次性完成，职责更清晰） |
| **校准指标** | 无 | **新增**：ECE（Expected Calibration Error，10 bins）、平均/正确/错误置信度、置信度间隙 |
| **潜在空间分析** | Silhouette + DB Index（仅打印） | **+ 返回结构化指标**：silhouette（cosine 口径）、类间/类内距离、分离比 |
| **多模态有效性探针** | 无 | **新增**：LogisticRegression + StratifiedKFold 分别探测 sMRI / FC / 融合特征 → `fusion_gain = fusion_acc − max(smri_acc, fc_acc)` |
| **CSV 输出** | 分类性能 + 泛化能力（2 个） | **+ 潜在空间可分性 + 多模态有效性 + 预测可依赖性（共 5 个）** |
| **结果可视化** | 无 | **新增 `test_img/`** — 7 张图（ROC、t-SNE、混淆矩阵、最佳轮次、早/晚融合、分类性能、训练总览） |
| **文件结构** | 7 个模块（encoders/fusion/proto/sampler 分离） | **整合为 5 个模块**：`models.py`（所有编码器+融合+头+增强）、`trainer.py`（训练+指标+采样器）、`domain.py`、`losses.py`、`main.py` |

### 设计理念变化总结

1. **图前先做时间 CNN** — 1D CNN 先学习每个 ROI 自身的时间动力学，GAT 再建模 ROI 间交互，解耦"单个脑区信号长什么样"与"脑区之间如何交互"。
2. **固定长度插值** — 所有 ROI 序列统一重采样到 T=150（替代 padding），消除长度差异同时保留信号形状，支持 batch CNN 处理。
3. **可靠性感知评估** — ECE + 置信度指标量化预测的可信程度，而不只是准确率。
4. **多模态有效性探针** — Logistic 探针单独评估每个模态的贡献并计算融合增益，证明融合确实优于最优单模态。
5. **代码整合** — 合并相关模块降低导入复杂度，架构不变。

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
  │  ROI BN (B,246,150) ──→ TemporalEncoder (1D CNN) ──→ GAT (2 层) ──→ z_functional (B,64)  │
  │                           （单 ROI 时间动力学）        （ROI 间消息传递）                  │
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
| `SMRIEncoder3D` | `models.py` | 3D CNN 编码 sMRI 体素数据 |
| `MorphEncoder` | `models.py` | 标量注意力池化处理 246 ROI × 6 形态学特征 |
| `TemporalEncoder` | `models.py` | **新增**：共享权重 1D CNN（kernel 7/5）提取单个 ROI 时间动力学 |
| `GATEncoder` | `models.py` | 2 层 GAT — TemporalEncoder 输出 → 图消息传递 |
| `StructuralGate` | `models.py` | 双门控融合 + 余弦相似度冲突回退 |
| `DiseaseFusion` | `models.py` | 3 个疾病 token + sigmoid 模态混合 + token_bias |
| `BrainDiseaseModel` | `models.py` | 顶层模型 |
| `DiseasePrototypeHead` | `models.py` | 可学习类别原型（并入 models.py） |
| `SMRIAugment` | `models.py` | sMRI 高斯噪声增强 |
| `GradientReversalLayer` | `domain.py` | GRL 梯度反转层，用于域对抗训练 |
| `DomainDiscriminator` | `domain.py` | CDAN 条件域判别器 |
| `HCDomainDiscriminator` | `domain.py` | HC-only 线性判别器 |
| `DomainWrapper` | `domain.py` | 为数据集样本添加域标签和健康标签 |
| `CombatMorphWrapper` | `domain.py` | ComBat 站点校正后的形态学特征 |
| `CombatFcWrapper` | `domain.py` | ComBat 校正后的 FC 矩阵（替换 ROI TS） |
| `SupConLoss` | `losses.py` | 监督对比损失 |
| `orthogonality_loss` | `losses.py` | z_structure ⟂ z_functional 正交约束 |
| `mmd_loss` / `consistency_loss` | `losses.py` | 域对齐损失 |
| `prototype_loss` / `token_diversity_loss` | `losses.py` | 原型锚定 + token 去相关 |
| `train_one_epoch` / `validate` | `trainer.py` | 训练循环 + 指标 + ECE 校准 |
| `compute_group_distances` | `trainer.py` | 潜在空间可分性 + 多模态有效性探针 |
| `BalancedBatchSampler` | `trainer.py` | 每 batch 固定各类样本数，少数类过采样 |

## 评估指标（v0.3 新增）

### 分类指标（每 epoch，CSV）

accuracy、macro-F1、weighted-F1、macro-AUC、precision/recall（macro + 各类）

### 预测可依赖性（`预测可依赖性.csv`）

| 指标 | 含义 |
|------|------|
| ECE | 10 bins 期望校准误差（↓ 好） |
| mean_confidence | 平均 max-softmax 置信度 |
| correct / wrong confidence | 正确/错误预测的平均置信度 |
| confidence_gap | 正确置信度 − 错误置信度（↑ 好） |

### 潜在空间可分性（`潜在空间可分性.csv`）

| 指标 | 含义 |
|------|------|
| latent_silhouette | z_disease 上的 Silhouette（cosine 口径，↑ 好） |
| latent_inter_distance | 类中心平均距离（1−cos，↑ 好） |
| latent_intra_distance | 类内样本-中心平均距离（↓ 好） |
| latent_separation_ratio | 类间 / 类内（↑ 好） |

### 多模态有效性（`多模态有效性.csv`）

对每个隐空间做 LogisticRegression 探针（StratifiedKFold，3 折）：

| 指标 | 含义 |
|------|------|
| smri_acc | 仅 z_structure 的探针准确率 |
| fc_acc | 仅 z_functional 的探针准确率 |
| fusion_acc | z_disease（融合后）的探针准确率 |
| fusion_gain | fusion_acc − max(smri_acc, fc_acc) — **融合有效性的直接证据** |

## 结果可视化（`test_img/`）

```
test_img/
├── Overview of Model Training.png    # 训练过程总览
├── Best round score.png              # 最佳轮次得分
├── Classification performance.png    # 分类性能
├── Confusion Matrix.png              # 混淆矩阵
├── ROC.png                           # ROC 曲线
├── t-SNE.png                         # 潜在空间 t-SNE 可视化
└── Early vs Late.png                 # 早融合 vs 晚融合对比
```

## 文件结构

```
2026.8.7_MODEL_ver0.3/
├── domain.py        # GRL、判别器、数据集/ComBat 包装器（原 domain_modules.py）
├── losses.py        # SupCon、正交、MMD、一致性、prototype、token 多样性
├── main.py          # 训练入口
├── models.py        # 全部编码器 + 融合 + BrainDiseaseModel + DiseasePrototypeHead + SMRIAugment
├── test_img/        # 测试结果可视化（7 张图）
└── trainer.py       # 训练/验证、指标、ECE、可分性探针、BalancedBatchSampler
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
| ROI BN | `(246, 150)` | Brainnetome 图谱 BOLD 时间序列（插值到 T=150） |

标签映射：`0 = HC`, `1 = Schizophrenia`, `2 = ADHD`

## 快速开始

### 环境依赖

- Python ≥ 3.8
- PyTorch ≥ 1.12
- NumPy, scikit-learn

### 训练

```bash
# 默认参数
python -m 2026.8.7_MODEL_ver0.3.main

# 使用 ComBat 站点校正（推荐）
python -m 2026.8.7_MODEL_ver0.3.main --combat_morph --combat_fc

# 完整自定义训练
python -m 2026.8.7_MODEL_ver0.3.main \
    --epochs 150 \
    --batch_size 24 \
    --t_target 150 \
    --lr 0.001 \
    --latent_dim 64 \
    --dropout 0.3 \
    --lambda_contrast 0.1 \
    --lambda_ortho 0.05 \
    --lambda_proto 0.5 \
    --lambda_hc_domain 0.1 \
    --combat_morph \
    --combat_fc \
    --seed 42
```

### 主要超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 24 | 批次大小（÷3 类 = 每类 8） |
| `--t_target` | 150 | **新增**：ROI 时间序列插值目标长度 |
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
- `best_prototypes.pt` — 学习到的疾病原型
- `history.json` — 训练历史
- `分类性能.csv` — 分类指标
- `泛化能力.csv` — 泛化间隙追踪
- `潜在空间可分性.csv` — 潜在空间可分性指标
- `多模态有效性.csv` — 多模态有效性探针 + 融合增益
- `预测可依赖性.csv` — ECE + 置信度校准指标

---

*Version 0.3 — 2026.8.7*

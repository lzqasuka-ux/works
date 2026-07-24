"""7.24_MODEL0.1 — 多模态脑疾病分类模型

结构：
    augmentation  — 数据增强
    encoders      — sMRI / FC / Morph 编码器
    fusion        — StructuralGate / FunctionalAttention / LearnableFC / DiseaseFusion
    model         — BrainDiseaseModel（主模型）
    losses        — SupConLoss
    trainer       — train_one_epoch / validate / compute_metrics
    main          — 训练入口
"""

from .model import BrainDiseaseModel
from .losses import SupConLoss
from .augmentation import SMRIAugment, FCAugment, smri_aug
from .encoders import SMRIEncoder3D, FCEncoder2D, MorphEncoder
from .fusion import StructuralGate, FunctionalAttention, LearnableFC, DiseaseFusion
from .trainer import train_one_epoch, validate, compute_metrics

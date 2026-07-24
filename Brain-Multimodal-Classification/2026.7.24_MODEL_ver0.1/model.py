"""主模型 BrainDiseaseModel：组合所有编码器与融合模块"""

import torch.nn as nn

from .encoders import SMRIEncoder3D, FCEncoder2D, MorphEncoder
from .fusion import StructuralGate, FunctionalAttention, LearnableFC, DiseaseFusion


class BrainDiseaseModel(nn.Module):
    """多模态脑疾病分类模型。

    输入：
        - sMRI 体素 (B, 1, D, H, W)
        - sMRI 形态学特征 (B, N_rois, N_features)
        - ROI 时间序列 BN 图谱 (B, N_rois, T_bn)
        - ROI 时间序列 AAL 图谱 (B, N_rois, T_aal)

    流程：
        sMRI ──→ SMRIEncoder3D ──→ z_voxel ──┐
        morph ─→ MorphEncoder   ──→ z_morph ──┤─ StructuralGate ──→ z_structure ──┐
        roi_bn ─→ LearnableFC → FCEncoder2D → z_bn ──┤                           │
        roi_aal → LearnableFC → FCEncoder2D → z_aal ──┤─ FunctionalAttention ──→ z_functional ──┤
                                                                                                   │
                               z_structure + z_functional ──→ DiseaseFusion ──→ z_disease ──→ Classifier
    """

    def __init__(self, num_classes=3, latent_dim=64, dropout=0.3,
                 n_timepoints_bn=200, n_timepoints_aal=200):
        super().__init__()
        self.sMRI_encoder = SMRIEncoder3D(latent_dim)
        self.morph_encoder = MorphEncoder(n_rois=246, n_features=6, latent_dim=latent_dim)
        self.fc_encoder_bn = FCEncoder2D(latent_dim)
        self.fc_encoder_aal = FCEncoder2D(latent_dim)
        self.norm_sMRI = nn.LayerNorm(latent_dim)
        self.norm_morph = nn.LayerNorm(latent_dim)
        self.norm_fc_bn = nn.LayerNorm(latent_dim)
        self.norm_fc_aal = nn.LayerNorm(latent_dim)

        # 可学习 FC 模块
        self.learnable_fc_bn = LearnableFC(n_timepoints=n_timepoints_bn, d_model=latent_dim)
        self.learnable_fc_aal = LearnableFC(n_timepoints=n_timepoints_aal, d_model=latent_dim)

        # 融合模块
        self.struct_fusion = StructuralGate(latent_dim)
        self.func_fusion = FunctionalAttention(latent_dim)
        self.disease_fusion = DiseaseFusion(latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.dropout = nn.Dropout(dropout)  # 编码器输出也过 dropout

    def forward(self, smri, smri_morph, roi_bn, roi_aal):
        # sMRI 分支
        z_voxel = self.dropout(self.norm_sMRI(self.sMRI_encoder(smri)))
        z_morph = self.dropout(self.norm_morph(self.morph_encoder(smri_morph)))
        z_structure = self.struct_fusion(z_voxel, z_morph)

        # 功能分支：ROI TS → LearnableFC → FC Encoder
        fc_bn = self.learnable_fc_bn(roi_bn).unsqueeze(1)     # (B, 1, N, N)
        fc_aal = self.learnable_fc_aal(roi_aal).unsqueeze(1)   # (B, 1, N, N)
        z_bn = self.dropout(self.norm_fc_bn(self.fc_encoder_bn(fc_bn)))
        z_aal = self.dropout(self.norm_fc_aal(self.fc_encoder_aal(fc_aal)))

        z_functional = self.func_fusion(z_bn, z_aal)
        z_disease = self.disease_fusion(z_structure, z_functional)

        logits = self.classifier(z_disease)
        return logits, z_disease, None

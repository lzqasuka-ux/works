"""BrainDiseaseModel — multimodal fusion classifier"""
import torch
import torch.nn as nn
from .encoders import SMRIEncoder3D, FCEncoder2D


class BrainDiseaseModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=32):
        super().__init__()
        self.sMRI_encoder = SMRIEncoder3D(latent_dim)
        self.fc_encoder = FCEncoder2D(latent_dim)
        # 防止两个模态尺度不同
        self.norm_sMRI = nn.LayerNorm(latent_dim)
        self.norm_FC = nn.LayerNorm(latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, smri, fc):
        z_smri = self.norm_sMRI(self.sMRI_encoder(smri))
        z_fc = self.norm_FC(self.fc_encoder(fc))
        z_shared = torch.cat([z_smri, z_fc], dim=1)
        logits = self.classifier(z_shared)
        return logits, z_shared
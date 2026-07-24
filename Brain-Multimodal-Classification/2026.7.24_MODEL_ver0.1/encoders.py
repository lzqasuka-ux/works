"""编码器模块：sMRI 3D 编码器、FC 2D 编码器、形态学编码器"""

import torch
import torch.nn as nn


class SMRIEncoder3D(nn.Module):
    """3D 卷积编码器，处理 sMRI 体素数据"""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1), nn.BatchNorm3d(8), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 4 * 4 * 4, latent_dim),
            nn.BatchNorm1d(latent_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.features(x))


class FCEncoder2D(nn.Module):
    """2D 卷积编码器，处理功能连接矩阵"""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 8 * 8, latent_dim),
            nn.BatchNorm1d(latent_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.features(x))


class MorphEncoder(nn.Module):
    """形态学特征编码器：ROI-wise MLP + MultiheadAttention 池化"""
    def __init__(self, n_rois=246, n_features=6, latent_dim=64):
        super().__init__()
        self.roi_mlp = nn.Sequential(
            nn.Linear(n_features, latent_dim), nn.ReLU(inplace=True), nn.LayerNorm(latent_dim),
        )
        self.attn_pool = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)
        self.query_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        B = x.size(0)
        z = self.roi_mlp(x)                                    # (B, N, latent_dim)
        query = self.query_token.expand(B, -1, -1)              # (B, 1, latent_dim)
        z_attn, _ = self.attn_pool(query=query, key=z, value=z)
        z_attn = z_attn.squeeze(1)                              # (B, latent_dim)
        z_attn = self.norm(z_attn + z.mean(dim=1))              # 残差连接均值池化
        return z_attn

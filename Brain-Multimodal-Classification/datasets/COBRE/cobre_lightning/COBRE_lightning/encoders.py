"""SMRI 3D CNN and FC 2D CNN encoders (lightweight, no BatchNorm)"""
import torch
import torch.nn as nn


class SMRIEncoder3D(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.features(x))


class FCEncoder2D(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.features(x))
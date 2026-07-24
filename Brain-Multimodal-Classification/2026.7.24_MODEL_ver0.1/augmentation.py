"""数据增强模块"""

import torch


class SMRIAugment:
    """sMRI 高斯噪声增强"""
    def __call__(self, x):
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * 0.005
        return x


class FCAugment:
    """FC 矩阵增强：高斯噪声 + 随机 dropout"""
    def __init__(self, noise_std=0.005, drop_prob=0.02):
        self.noise_std = noise_std
        self.drop_prob = drop_prob

    def __call__(self, x):
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.drop_prob > 0:
            mask = (torch.rand_like(x) > self.drop_prob).float()
            x = x * mask
        return x


# 全局 sMRI 增强实例（供 trainer 使用）
smri_aug = SMRIAugment()

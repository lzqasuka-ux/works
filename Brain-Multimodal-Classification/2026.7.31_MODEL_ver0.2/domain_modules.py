"""Auto-split from train_all_in_one - backup2.py"""
import torch
import torch.nn as nn
import numpy as np

class GradientReversalLayer(torch.autograd.Function):
    """GRL：forward 不变，backward 梯度乘 -alpha 翻转"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class HCDomainDiscriminator(nn.Module):
    """HC-only 域判别器：极简，只在 HC 上训练"""
    def __init__(self, in_dim=192, num_domains=2):
        super().__init__()
        self.net = nn.Linear(in_dim, num_domains)  # 386 参数

    def forward(self, z, alpha=1.0):
        z = GradientReversalLayer.apply(z, alpha)
        return self.net(z)

class DomainDiscriminator(nn.Module):
    """CDAN 条件域判别器：GRL(z) ⊗ softmax(class_logits).detach()"""
    def __init__(self, dim=64, num_classes=3):
        super().__init__()
        in_dim = dim * num_classes  # 64*3=192
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
    def forward(self, z, class_logits, alpha=1.0):
        # z: (B, dim), class_logits: (B, num_classes)
        prob = torch.softmax(class_logits.detach(), dim=-1)    # (B, K)
        z_grl = GradientReversalLayer.apply(z, alpha)          # (B, dim)
        # outer product + flatten
        joint = torch.einsum('bd,bk->bdk', z_grl, prob)        # (B, dim, K)
        joint = joint.reshape(z.size(0), -1)                    # (B, dim*K)
        return self.net(joint)

# Domain wrapper：不动数据集代码，只给每个样本打数据集 ID
class DomainWrapper:
    def __init__(self, dataset, domain_id):
        self.dataset = dataset
        self.domain_id = domain_id
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["domain"] = self.domain_id
        item["is_hc"] = (item["label"] == 0)
        return item
    def __len__(self):
        return len(self.dataset)
    def __getattr__(self, name):
        return getattr(self.dataset, name)

class CombatMorphWrapper:
    """替换 sMRI_morph 为 ComBat 校正后的特征（不动数据集代码）"""
    def __init__(self, dataset, combat_morph):
        self.dataset = dataset
        self.combat_morph = combat_morph  # (n, 246, 6) numpy
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["sMRI_morph"] = torch.from_numpy(self.combat_morph[idx].astype(np.float32))
        return item
    def __len__(self):
        return len(self.dataset)
    def __getattr__(self, name):
        return getattr(self.dataset, name)

class CombatFcWrapper:
    """替换 ROI_bn 为 ComBat 校正后的 FC 矩阵（不动数据集代码）"""
    def __init__(self, dataset, combat_fc):
        self.dataset = dataset
        self.combat_fc = combat_fc  # (n, 246, 246) numpy
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["ROI_bn"] = torch.from_numpy(self.combat_fc[idx].astype(np.float32))
        return item
    def __len__(self):
        return len(self.dataset)
    def __getattr__(self, name):
        return getattr(self.dataset, name)

# ============================
# Disease Prototype Head
# ============================

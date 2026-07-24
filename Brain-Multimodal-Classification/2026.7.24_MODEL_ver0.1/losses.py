"""损失函数：Supervised Contrastive Loss"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    让同类别样本在隐空间靠近，不同类别远离。
    单类别样本（无正样本对）自动跳过，不贡献 loss。
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, D), labels: (B,)
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        features = nn.functional.normalize(features, dim=1)
        sim = features @ features.T / self.temperature            # (B, B)

        # 正样本 mask：同类 & 非自身
        pos_mask = (labels[:, None] == labels[None, :]).float()  # (B, B)
        pos_mask.fill_diagonal_(0)

        # 数值稳定: 减去每行最大值
        sim = sim - sim.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim)

        # 分母：所有非自身样本的 exp 之和
        exp_sim_no_diag = exp_sim * (1 - torch.eye(B, device=sim.device))
        all_sum = exp_sim_no_diag.sum(dim=1).clamp(min=1e-8)    # (B,)

        # 分子：正样本的 exp 之和
        pos_sum = (exp_sim * pos_mask).sum(dim=1)                # (B,)

        # 只对至少有 1 个正样本的 anchor 计算 loss
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 按正样本数归一化（SupCon 标准做法）
        n_pos = pos_mask.sum(dim=1)[valid].clamp(min=1)          # (n_valid,)
        loss = -(torch.log(pos_sum[valid] / all_sum[valid]) / n_pos).mean()
        return loss

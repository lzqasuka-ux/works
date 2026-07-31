"""Auto-split from train_all_in_one - backup2.py"""
import torch
import torch.nn as nn
import numpy as np

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

# ---------------------------------------------------------------------------
# Domain-aware Supervised Contrastive Loss
# ---------------------------------------------------------------------------
class DomainAwareSupConLoss(nn.Module):
    """跨域对比损失：同病同域=强(1.0)，同病跨域=弱(0.4)"""

    def __init__(self, temperature=0.07, same_weight=1.0, cross_weight=0.4):
        super().__init__()
        self.temperature = temperature
        self.same_weight = same_weight     # 同病同域 = 强
        self.cross_weight = cross_weight   # 同病跨域 = 弱

    def forward(self, features, labels, domains):
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=features.device, requires_grad=True), {}
        domains = domains.to(features.device)

        features = nn.functional.normalize(features, dim=1)
        sim = features @ features.T / self.temperature

        # 正样本权重矩阵
        same_dis = (labels[:, None] == labels[None, :]).float()
        same_dis.fill_diagonal_(0)
        cross_domain = same_dis * (domains[:, None] != domains[None, :]).float()
        same_domain  = same_dis * (domains[:, None] == domains[None, :]).float()
        pos_weight = self.same_weight * same_domain + self.cross_weight * cross_domain  # (B,B)

        # 数值稳定
        sim = sim - sim.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim)

        # 分母：所有非自身
        denom = exp_sim.sum(dim=1) - exp_sim.diag().clamp(min=1e-8)

        # 分子：加权正样本
        numer = (exp_sim * pos_weight).sum(dim=1)
        eff_n = pos_weight.sum(dim=1).clamp(min=1e-8)

        valid = pos_weight.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True), {}

        loss = -(torch.log(numer[valid] / denom[valid].clamp(min=1e-8)) / eff_n[valid]).mean()

        stats = {
            "cross_domain_pairs": cross_domain.sum().item(),
            "same_domain_pairs": same_domain.sum().item(),
            "total_valid": valid.sum().item(),
        }
        return loss, stats

# ---------------------------------------------------------------------------
# Orthogonality Loss
# ---------------------------------------------------------------------------
def orthogonality_loss(z_s, z_f):
    """鼓励结构和功能隐向量承载互补而非冗余的信息。

    如果 z_s 和 z_f 高度相关，模型可能是用两个模态学同一件事。
    正交性约束推动它们各自承载不同因素，适配 "结构正常但功能异常" 等解耦模式。
    """
    z_s = torch.nn.functional.normalize(z_s, dim=-1)
    z_f = torch.nn.functional.normalize(z_f, dim=-1)
    return (z_s * z_f).sum(dim=-1).pow(2).mean()

def mmd_loss(z, domains, labels=None):
    """Per-class MMD：按疾病类别分别对齐跨域样本。无标签时退化为全局 MMD"""
    domains = domains.to(z.device)
    if labels is not None:
        labels = labels.to(z.device)
    if labels is None:
        mask0 = (domains == 0)
        mask1 = (domains == 1)
        if mask0.sum() == 0 or mask1.sum() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        return (z[mask0].mean(0) - z[mask1].mean(0)).pow(2).sum() / z.size(-1)

    loss = torch.tensor(0.0, device=z.device)
    D = z.size(-1)
    for cls in labels.unique():
        mask = (labels == cls)
        z_c, d_c = z[mask], domains[mask]
        m0, m1 = (d_c == 0), (d_c == 1)
        if m0.sum() > 0 and m1.sum() > 0:
            loss = loss + (z_c[m0].mean(0) - z_c[m1].mean(0)).pow(2).sum()
    return loss / D

def consistency_loss(z, labels, domains):
    """HC样本到同域HC和跨域HC的距离应该一致。z:(B,D)"""
    domains = domains.to(z.device)
    hc = (labels == 0)
    if hc.sum() < 2:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    z_hc, d_hc = z[hc], domains[hc]
    loss, n = 0.0, 0
    for i in range(len(z_hc)):
        others = torch.arange(len(z_hc), device=z.device) != i
        same = (d_hc == d_hc[i]) & others
        cross = (d_hc != d_hc[i]) & others
        if same.sum() == 0 or cross.sum() == 0: continue
        d_s = (z_hc[i] - z_hc[same]).pow(2).sum(dim=-1).mean()
        d_c = (z_hc[i] - z_hc[cross]).pow(2).sum(dim=-1).mean()
        loss = loss + (d_s - d_c).pow(2)
        n += 1
    return loss / max(n, 1)

def prototype_loss(tokens, labels, prototypes):
    """L2 距离拉样本 token 到原型，梯度比余弦更猛"""
    t = torch.nn.functional.normalize(tokens, dim=-1)
    p = torch.nn.functional.normalize(prototypes, dim=-1)
    t_class = t[range(t.size(0)), labels]                        # (B, D)
    p_class = p[labels]                                           # (B, D)
    return (t_class - p_class).pow(2).sum(dim=-1).mean()

def token_diversity_loss(tokens):
    """鼓励 3 个 token 输出互补——同类样本的 token_i 和 token_j 应该不同"""
    t = torch.nn.functional.normalize(tokens, dim=-1)            # (B, K, D)
    K = t.size(1)
    loss = 0.0
    for i in range(K):
        for j in range(i+1, K):
            loss += (t[:, i] * t[:, j]).sum(dim=-1).abs().mean()  # cos 越接近 0 越好
    return loss / (K * (K-1) / 2)

def proto_separation_loss(prototypes):
    """鼓励原型之间相互正交：对角线=1，非对角线→0或负"""
    p = torch.nn.functional.normalize(prototypes, dim=-1)
    cos_mat = p @ p.T                                      # (K, K)
    # 非对角线上取 mean squared positive cosine
    K = cos_mat.size(0)
    mask = ~torch.eye(K, dtype=torch.bool, device=cos_mat.device)
    return cos_mat[mask].clamp(min=0).pow(2).mean()

# ---------------------------------------------------------------------------
# 域对抗：梯度反转层 + 域判别器
# ---------------------------------------------------------------------------

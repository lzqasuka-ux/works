"""跨模态融合模块：StructuralGate, FunctionalAttention, LearnableFC, DiseaseFusion"""

import torch
import torch.nn as nn


# ============================
# Structural Gate Fusion (z_voxel ↔ z_morph)
# ============================
class StructuralGate(nn.Module):
    """Dual Structural Gate：两个独立门控分别加权体素和形态特征"""

    def __init__(self, dim=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),  # 输出 2×dim，前 dim 为 g_v，后 dim 为 g_m
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_voxel, z_morph):
        # z_voxel: (B, dim), z_morph: (B, dim)
        gates = self.gate(torch.cat([z_voxel, z_morph], dim=-1))  # (B, 2*dim)
        g_v, g_m = gates.chunk(2, dim=-1)                          # 各 (B, dim)
        z_structure = g_v * z_voxel + g_m * z_morph               # (B, dim)
        return self.norm(z_structure)


# ============================
# Atlas Adaptive Fusion (z_bn ↔ z_aal)
# ============================
class FunctionalAttention(nn.Module):
    """Atlas Adaptive Fusion：softmax 加权 + 残差投影融合两个脑图谱 FC 特征。

    设计动机：BN 和 AAL 是固定的两个图谱（而非变长序列），用 softmax(MLP)
    比 MHA 更直接、参数更少、且天然保证 α_BN + α_AAL = 1。
    """

    def __init__(self, dim=64):
        super().__init__()
        # 标量权重 MLP: concat(B,2*dim) → alpha(B,2)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
        )
        # 残差投影: concat → 非线性混合补偿
        self.res_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_bn, z_aal):
        # z_bn: (B, dim), z_aal: (B, dim)
        concat = torch.cat([z_bn, z_aal], dim=-1)          # (B, 2*dim)

        # 每样本标量权重，softmax 使得 α_bn + α_aal = 1
        alpha = torch.softmax(self.alpha_mlp(concat), dim=-1)  # (B, 2)
        alpha_bn = alpha[:, 0:1]                                # (B, 1)
        alpha_aal = alpha[:, 1:2]                               # (B, 1)

        # 加权融合
        z_functional = alpha_bn * z_bn + alpha_aal * z_aal  # (B, dim)

        # 残差连接：concat 的非线性投影
        z_functional = z_functional + self.res_proj(concat) # (B, dim)

        return self.norm(z_functional)


# ============================
# Learnable FC Module (ROI time series → FC matrix)
# ============================
class LearnableFC(nn.Module):
    """可学习功能连接矩阵生成器。

    ROI time series (B, N, T) → Linear(T→d) → Q, K 注意力 → 对称 FC 矩阵 (B, N, N)
    """

    def __init__(self, n_timepoints, d_model=64):
        super().__init__()
        self.Q = nn.Linear(n_timepoints, d_model, bias=False)
        self.K = nn.Linear(n_timepoints, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N, T) — N 个脑区, T 个时间点
        B, N, T = x.shape

        # 投影: (B, N, T) → (B, N, d)
        q = self.Q(x)  # (B, N, d)
        k = self.K(x)  # (B, N, d)
        q = self.ln(q)
        k = self.ln(k)

        # 缩放点积注意力
        d = q.size(-1)
        fc = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (d ** 0.5), dim=-1)  # (B, N, N)

        # 对称化
        fc = (fc + fc.transpose(1, 2)) / 2.0

        return fc


# ============================
# Disease Token Fusion (z_structure + z_functional → z_disease)
# ============================
class DiseaseFusion(nn.Module):
    """Disease Token 对结构+功能特征做 Cross-Attention，带模态嵌入。

    为什么需要 modality embedding：
    z_structure 和 z_functional 经过多层融合后已压缩为通用隐向量 (B,64)，
    disease token 的 Cross-Attention 无法区分哪个来自结构、哪个来自功能。
    添加可学习的模态嵌入 (structure_embedding / functional_embedding) 让
    Key/Value token 携带模态身份信息，帮助 disease query 学习到更精确的
    跨模态疾病表征。
    """

    def __init__(self, dim=64):
        super().__init__()
        self.disease_token = nn.Parameter(torch.randn(1, dim) * 0.02)

        # 模态身份嵌入：标识 token 来自结构还是功能
        self.structure_embedding = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.functional_embedding = nn.Parameter(torch.randn(1, dim) * 0.02)

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, z_s, z_f):
        # z_s: (B, dim), z_f: (B, dim)
        B = z_s.size(0)

        # Disease token → query
        d = self.disease_token.expand(B, -1)       # (B, dim)
        q = self.query(d).unsqueeze(1)             # (B, 1, dim)

        # 模态嵌入注入: 为每个 token 打上结构/功能身份标签
        s_token = z_s + self.structure_embedding.expand(B, -1)   # (B, dim)
        f_token = z_f + self.functional_embedding.expand(B, -1)   # (B, dim)
        kv_tokens = torch.stack([s_token, f_token], dim=1)        # (B, 2, dim)

        k = self.key(kv_tokens)                      # (B, 2, dim)
        v = self.value(kv_tokens)                    # (B, 2, dim)

        # Scaled dot-product attention
        scale = k.size(-1) ** 0.5
        att = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / scale, dim=-1)  # (B, 1, 2)
        out = torch.matmul(att, v).squeeze(1)  # (B, dim)

        return self.norm(out + d)  # 残差连接 disease token

"""models.py — 模型定义模块 (SMRIAugment, 编码器, DiseaseFusion, BrainDiseaseModel, DiseasePrototypeHead)"""
import torch
import torch.nn as nn

class SMRIAugment:
    def __call__(self, x):
        if torch.rand(1).item() < 0.5:
            x = x + torch.randn_like(x) * 0.005
        return x

smri_aug = SMRIAugment()

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class SMRIEncoder3D(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1), nn.BatchNorm3d(8), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4,4,4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(16*4*4*4, latent_dim),
            nn.BatchNorm1d(latent_dim), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.fc(self.features(x))

# ============================
# GAT Encoder (FC matrix → graph message passing)
# ============================


class GATLayer(nn.Module):
    """单层 Graph Attention，用 FC 矩阵作为边权重调制注意力系数。

    e_ij = LeakyReLU(a_l·Wh_i + a_r·Wh_j) × FC_ij
    α_ij = softmax_j(e_ij)
    h_i' = Σ_j α_ij · Wh_j
    """
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.head_dim = out_dim // n_heads if concat else out_dim

        self.W  = nn.Linear(in_dim, n_heads * self.head_dim, bias=False)
        self.a_l = nn.Linear(self.head_dim, 1, bias=False)
        self.a_r = nn.Linear(self.head_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x, edge_weight=None):
        # x: (B, N, in_dim), edge_weight: 可选，已废弃
        B, N, _ = x.shape

        Wh = self.W(x).view(B, N, self.n_heads, self.head_dim)  # (B, N, H, D)

        # 标量注意力分数
        attn_l = self.a_l(Wh).squeeze(-1)                        # (B, N, H)
        attn_r = self.a_r(Wh).squeeze(-1)                        # (B, N, H)
        e = attn_l.unsqueeze(2) + attn_r.unsqueeze(1)            # (B, N, N, H)
        e = self.leaky_relu(e)

        # 不再用外部 FC 矩阵调制——attention 权重完全由 GAT 自学习

        alpha = torch.softmax(e, dim=2)                           # (B, N, N, H)
        alpha = self.dropout_attn(alpha)

        # 聚合: 多头 batch matmul
        alpha_t = alpha.permute(0, 3, 1, 2).reshape(B * self.n_heads, N, N)
        Wh_t = Wh.permute(0, 2, 1, 3).reshape(B * self.n_heads, N, self.head_dim)
        out = torch.bmm(alpha_t, Wh_t)                            # (B*H, N, D)
        out = out.view(B, self.n_heads, N, self.head_dim).permute(0, 2, 1, 3)  # (B, N, H, D)

        if self.concat:
            out = out.reshape(B, N, self.n_heads * self.head_dim)  # (B, N, H*D)
        else:
            out = out.mean(dim=2)                                   # (B, N, D)
        return out


class TemporalEncoder(nn.Module):
    """时间 CNN：每个 ROI 独立提取时间动力学模式。

    输入: (B, N, T) → 输出: (B, N, 64)
    所有 ROI 共享权重，等价于把 246 条时间序列各自过 CNN。
    """
    def __init__(self, hidden=32, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(hidden, out_dim, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # LayerNorm 在特征维归一化，保留 ROI 之间的尺度差异
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (B, N, T)
        B, N, T = x.shape
        x = x.reshape(B * N, 1, T)                    # (B*N, 1, T)
        x = torch.relu(self.conv1(x))                 # (B*N, 32, T)
        x = torch.relu(self.conv2(x))                 # (B*N, 64, T)
        x = self.pool(x).squeeze(-1)                   # (B*N, 64)
        x = x.reshape(B, N, -1)                        # (B, N, 64)
        return self.norm(x)                            # 特征维归一化


class GATEncoder(nn.Module):
    """2 层 GAT + 残差，FC 矩阵既充当节点特征又充当边权重。

    输入: ROI 时间序列 (B, N, T) → TemporalEncoder → GAT
    输出: (B, out_dim)
    """
    def __init__(self, n_nodes=246, n_timepoints=200, hidden_dim=64,
                 out_dim=64, n_heads=4, dropout=0.3, use_fc_input=False):
        super().__init__()
        self.use_fc_input = use_fc_input
        if use_fc_input:
            self.fc_proj = nn.Linear(n_nodes, hidden_dim)
        else:
            self.temporal_encoder = TemporalEncoder(hidden=32, out_dim=hidden_dim)
        self.roi_embed = nn.Parameter(torch.randn(1, n_nodes, hidden_dim) * 0.1)
        self.node_norm = nn.LayerNorm(hidden_dim)     # 放大节点间差异
        self.node_dropout = nn.Dropout(0.2)           # 训练时随机抹掉部分节点特征
        self.gat1 = GATLayer(hidden_dim, hidden_dim, n_heads=n_heads,
                             dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GATLayer(hidden_dim, out_dim, n_heads=1,
                             dropout=dropout, concat=False)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: (B, N, T) or (B, N, N) — ROI TS or FC matrix
        if self.use_fc_input:
            x = self.fc_proj(x)             # (B, N, N) → (B, N, hidden)
        else:
            # ROI 嵌入提前加到输入，打破 246 节点趋同
            x = self.temporal_encoder(x + self.roi_embed.sum(dim=-1, keepdim=True))
        x = x + self.roi_embed              # (B, N, hidden)  再次强化节点身份
        x = self.node_norm(x)               # 放大差异
        x = self.node_dropout(x)            # 训练时打破对称性
        x = x + self.gat1(x)               # 第 1 层 GAT + 残差（attention 全自学习）
        x = self.norm1(x)
        x = x + self.gat2(x)               # 第 2 层 GAT + 残差
        x = self.norm2(x)
        x = x.mean(dim=1)                  # (B, out_dim)  全局平均池化
        return x


class MorphEncoder(nn.Module):
    """轻量注意力池化：每个脑区学一个标量重要性分数，加权平均。

    相比原来的 MHA + Query Token，参数量从 ~12k 降到 ~640，
    更适合 246 脑区 × 6 特征的低信息密度输入和几百样本的数据规模。
    """
    def __init__(self, n_rois=246, n_features=6, latent_dim=64):
        super().__init__()
        self.roi_mlp = nn.Sequential(
            nn.Linear(n_features, latent_dim), nn.ReLU(inplace=True), nn.LayerNorm(latent_dim),
        )
        # 标量注意力打分：每个脑区的 64 维向量 → 1 个重要性分数
        self.attn_score = nn.Linear(latent_dim, 1)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # x: (B, 246, 6)
        z = self.roi_mlp(x)                                # (B, 246, 64)

        # 标量注意力分数
        score = self.attn_score(z).squeeze(-1)              # (B, 246)
        weight = torch.softmax(score, dim=-1).unsqueeze(-1) # (B, 246, 1)

        # 加权池化 + 残差均值
        z_attn = (z * weight).sum(dim=1)                    # (B, 64)
        return self.norm(z_attn + z.mean(dim=1))

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
        concat = torch.cat([z_voxel, z_morph], dim=-1)
        gates = self.gate(concat)                                    # (B, 2*dim)
        g_v, g_m = gates.chunk(2, dim=-1)                            # 各 (B, dim)

        # ── 特征分歧感知：一致时信任门控，冲突时退化为简单平均 ──
        cos_sim = torch.nn.functional.cosine_similarity(
            z_voxel, z_morph, dim=-1)                                # (B,)
        trust = cos_sim.clamp(min=0).unsqueeze(-1)                   # (B, 1)
        # trust → 1: 两分支一致，用门控；trust → 0: 冲突，退化为平均
        gated = g_v * z_voxel + g_m * z_morph
        averaged = (z_voxel + z_morph) / 2
        z_structure = trust * gated + (1 - trust) * averaged         # (B, dim)

        return self.norm(z_structure)


# ============================
# Disease Token Fusion (z_structure + z_functional → z_disease)
# ============================


class DiseaseFusion(nn.Module):
    """Multi-Disease Token：每类疾病一个可学习 token，独立做 Cross-Attention。

    单个 disease token 需要同时编码 HC / SZ / ADHD 三种模式，表达能力受限。
    改为每个疾病一个 token 后，各自专注一种疾病模式，语义更解耦。

    z_structure + z_functional → num_tokens × Cross-Attention → concat → (B, num_tokens*dim)
    """

    def __init__(self, dim=64, num_tokens=3):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.disease_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)

        # 每 token 可学习的模态混合权重 (sigmoid → 不竞争)
        self.modal_weight = nn.Parameter(torch.zeros(num_tokens, 2))
        # 每 token 独立的模态投影 + 视角偏置
        self.token_bias = nn.Parameter(torch.randn(num_tokens, dim))
        self.values_s = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_tokens)])
        self.values_f = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_tokens)])

        self.norm = nn.LayerNorm(dim * num_tokens)

    def forward(self, z_s, z_f):
        # z_s: (B, dim), z_f: (B, dim)
        B = z_s.size(0)

        # Disease token 残差
        d = self.disease_tokens.unsqueeze(0).expand(B, -1, -1)        # (B, K, dim)

        # Per-token 模态混合权重
        w = torch.sigmoid(self.modal_weight)                         # (K, 2)
        w = w.unsqueeze(0).unsqueeze(2)                               # (1, K, 1, 2)

        # 各自投影 + 加权混合（每 token 独立投影）
        outs = []
        for k in range(self.num_tokens):
            v_s = self.values_s[k](z_s)                           # (B, dim)
            v_f = self.values_f[k](z_f)
            # token_bias 作为大尺度偏置直接加输出上
            out_k = w[0, k, 0, 0] * v_s + w[0, k, 0, 1] * v_f + self.token_bias[k]  # (B, dim)
            outs.append(out_k)
        out = torch.stack(outs, dim=1)                             # (B, K, dim)

        # ── 特征分歧感知 ──
        cos_sf = torch.nn.functional.cosine_similarity(
            z_s, z_f, dim=-1)                                          # (B,)
        trust = cos_sf.clamp(min=0).view(B, 1, 1).expand(-1, self.num_tokens, -1)  # (B, K, 1)
        avg = out.mean(dim=1, keepdim=True)                                          # (B, 1, dim)
        out = trust * out + (1 - trust) * avg                          # (B, K, dim)

        out_flat = out.reshape(B, self.num_tokens * self.dim)          # (B, K*dim)

        return self.norm(out_flat + d.reshape(B, self.num_tokens * self.dim)), \
               out  # (B, K, dim) 处理后的 tokens，带梯度


class BrainDiseaseModel(nn.Module):
    def __init__(self, num_classes=3, latent_dim=64, dropout=0.3,
                 n_timepoints_bn=200, use_fc_input=False):
        super().__init__()
        self.sMRI_encoder = SMRIEncoder3D(latent_dim)
        self.morph_encoder = MorphEncoder(n_rois=246, n_features=6, latent_dim=latent_dim)
        self.fc_encoder_bn = GATEncoder(n_nodes=246, n_timepoints=n_timepoints_bn,
                                         hidden_dim=latent_dim, out_dim=latent_dim,
                                         use_fc_input=use_fc_input)
        self.norm_sMRI = nn.LayerNorm(latent_dim)
        self.norm_morph = nn.LayerNorm(latent_dim)
        self.norm_fc_bn = nn.LayerNorm(latent_dim)

        # 融合模块
        self.struct_fusion = StructuralGate(latent_dim)
        self.disease_fusion = DiseaseFusion(latent_dim, num_tokens=num_classes)
        out_dim = latent_dim * num_classes

        self.classifier_proto = DiseasePrototypeHead(token_dim=latent_dim, num_classes=num_classes, temperature=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.dropout = nn.Dropout(dropout)  # 编码器输出也过 dropout

    def forward(self, smri, smri_morph, roi_bn):
        # sMRI 分支
        z_voxel = self.dropout(self.norm_sMRI(self.sMRI_encoder(smri)))
        z_morph = self.dropout(self.norm_morph(self.morph_encoder(smri_morph)))
        z_structure = self.struct_fusion(z_voxel, z_morph)

        # 功能分支：ROI TS → GAT Encoder (图消息传递，直接吃时间序列)
        z_functional = self.dropout(self.norm_fc_bn(self.fc_encoder_bn(roi_bn)))

        z_disease, disease_tokens = self.disease_fusion(z_structure, z_functional)

        logits = self.classifier(z_disease)           # MLP → CE
        return logits, z_disease, (z_structure, z_functional), disease_tokens

# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------


class DiseasePrototypeHead(nn.Module):
    """每 disease token 一对一匹配自己的原型，梯度直达 token 层面"""
    def __init__(self, token_dim=64, num_classes=3, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        proto = torch.randn(num_classes, token_dim) * 0.02
        proto = torch.nn.functional.normalize(proto, dim=-1)
        self.prototypes = nn.Parameter(proto)

    def forward(self, tokens):
        # tokens: (B, K, D) — 每个疾病 token 的 64 维输出
        t = torch.nn.functional.normalize(tokens, dim=-1)        # (B, K, D)
        p = torch.nn.functional.normalize(self.prototypes, dim=-1) # (K, D)
        # token_i ↔ proto_i  对角线得分
        logits = (t * p.unsqueeze(0)).sum(dim=-1)                # (B, K)
        return logits / self.temperature

# ---------------------------------------------------------------------------
# 训练/验证
# ---------------------------------------------------------------------------


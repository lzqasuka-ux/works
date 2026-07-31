"""Auto-split from train_all_in_one - backup2.py"""
import torch
import torch.nn as nn
import numpy as np

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

"""Auto-split from train_all_in_one - backup2.py"""
import torch
import torch.nn as nn
import numpy as np

class BalancedBatchSampler:
    """每 batch 固定各类样本数量，少数类过采样"""
    def __init__(self, labels, n_per_class=8, shuffle=True):
        self.n_per_class = n_per_class
        self.shuffle = shuffle
        self.class_indices = {}
        for c in np.unique(labels):
            self.class_indices[c] = np.where(labels == c)[0]
        # epoch 长度由最大的类决定，少样本类过采样
        self.n_batches = max(len(v) for v in self.class_indices.values()) // n_per_class

    def __iter__(self):
        n_needed = self.n_batches * self.n_per_class
        indices_per_class = {}
        for c, pool in self.class_indices.items():
            if len(pool) >= n_needed:
                chosen = pool.copy()
                if self.shuffle:
                    np.random.shuffle(chosen)
                chosen = chosen[:n_needed]
            else:
                chosen = np.random.choice(pool, size=n_needed, replace=True)
            indices_per_class[c] = chosen.reshape(-1, self.n_per_class)

        batches = []
        for b in range(self.n_batches):
            batch = np.concatenate([indices_per_class[c][b]
                                    for c in range(len(self.class_indices))])
            if self.shuffle:
                np.random.shuffle(batch)
            batches.append(batch.tolist())
        return iter(batches)

    def __len__(self):
        return self.n_batches

# ---------------------------------------------------------------------------
# 主训练入口
# ---------------------------------------------------------------------------

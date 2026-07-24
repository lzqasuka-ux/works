"""主训练入口：多模态多分类训练 (COBRE + ADHD-200)"""

import os
import sys
import json
import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

from .model import BrainDiseaseModel
from .losses import SupConLoss
from .trainer import train_one_epoch, validate


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | BS: {args.batch_size} | "
          f"LR: {args.lr} | Latent: {args.latent_dim}")

    # ---- 数据集导入 ----
    sys.path.insert(0, args.cobre_dir)
    sys.path.insert(0, args.adhd_dir)
    from cobre_dataset import COBREDataset
    from adhd200_dataset import ADHD200Dataset

    print("[Data] Loading COBRE ...")
    cobre_ds = COBREDataset(split="all", seed=args.seed, normalize_smri=True, fisher_z=True,
                             base_dir=args.cobre_dir)
    print("[Data] Loading ADHD-200 ...")
    adhd_ds = ADHD200Dataset(split="all", seed=args.seed, normalize_smri=True, fisher_z=True,
                              base_dir=args.adhd_dir)

    # Label mapping: ADHD-200 1→2 (ADHD)
    adhd_ds.labels = adhd_ds.labels.copy()
    adhd_ds.labels[adhd_ds.labels == 1] = 2

    combined_ds = ConcatDataset([cobre_ds, adhd_ds])
    all_labels = np.concatenate([cobre_ds.labels, adhd_ds.labels])
    print(f"[Data] 合并数据集: {len(combined_ds)} 样本 "
          f"(COBRE: {len(cobre_ds)} + ADHD-200: {len(adhd_ds)})")
    print(f"  类别分布: 0={np.sum(all_labels==0)}, "
          f"1={np.sum(all_labels==1)}, 2={np.sum(all_labels==2)}")

    # ---- Stratified split ----
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(all_labels):
        cls_idx = np.where(all_labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train:n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    test_idx = np.sort(np.concatenate(test_idx))
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # ---- 扫描全局 max T ----
    print("  Scanning max T ... ", end="", flush=True)
    global_max_T_bn = 0
    global_max_T_aal = 0
    for i in range(len(cobre_ds)):
        t = cobre_ds.get_roi_T(i)
        global_max_T_bn = max(global_max_T_bn, t)
        global_max_T_aal = max(global_max_T_aal, t)
    for i in range(len(adhd_ds)):
        t = adhd_ds.get_roi_T(i)
        global_max_T_bn = max(global_max_T_bn, t)
        global_max_T_aal = max(global_max_T_aal, t)
    print(f"BN={global_max_T_bn}, AAL={global_max_T_aal}")

    # ---- DataLoader ----
    def collate_fn(batch):
        out = {}
        for key in batch[0].keys():
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                # Pad variable-length ROI time series to global max T
                if vals[0].ndim == 2 and key.startswith("ROI"):
                    target_len = global_max_T_bn if key == "ROI_bn" else global_max_T_aal
                    padded = []
                    for v in vals:
                        if v.shape[1] < target_len:
                            v = torch.nn.functional.pad(v, (0, target_len - v.shape[1]))
                        padded.append(v)
                    out[key] = torch.stack(padded)
                else:
                    out[key] = torch.stack(vals)
        return out

    train_loader = DataLoader(Subset(combined_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(combined_ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(combined_ds, test_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ---- 模型 ----
    model = BrainDiseaseModel(num_classes=args.num_classes, latent_dim=args.latent_dim,
                              dropout=args.dropout,
                              n_timepoints_bn=global_max_T_bn,
                              n_timepoints_aal=global_max_T_aal).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数: {n_params:,} | "
          f"训练 batch: {len(train_loader)} | "
          f"验证 batch: {len(val_loader)} | 测试 batch: {len(test_loader)}")

    # ---- 优化器 & 调度器 ----
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_contrast = SupConLoss(temperature=0.07) if args.lambda_contrast > 0 else None
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                      eta_min=args.lr * 0.01)

    # ---- 训练循环 ----
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "per_class": []}
    best_acc, best_epoch = 0, 0
    patience_counter = 0
    total_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        print(f"\n── Epoch {epoch}/{args.epochs} ──")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                      criterion_contrast=criterion_contrast,
                                      lambda_contrast=args.lambda_contrast)
        val_loss, m = validate(model, val_loader, criterion, device, desc="Val")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(m["accuracy"])
        history["per_class"].append(m["per_class"])

        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        pc = m["per_class"]
        ep_time = time.time() - t_epoch
        total_time = time.time() - total_t0
        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {m['accuracy']:.3f} | "
              f"PC: {pc.get(0,0):.2f}/{pc.get(1,0):.2f}/{pc.get(2,0):.2f} | "
              f"LR: {lr:.2e} | {ep_time:.0f}s/epoch, total {total_time/60:.1f}min")

        if m["accuracy"] > best_acc:
            best_acc, best_epoch = m["accuracy"], epoch
            patience_counter = 0
            for old in glob.glob(args.save_path.replace('.pth', '_epoch*.pth')):
                os.remove(old)
            save_name = args.save_path.replace(
                '.pth',
                f'_epoch{epoch}_train{train_loss:.4f}_val{val_loss:.4f}_acc{m["accuracy"]:.4f}.pth')
            torch.save(model.state_dict(), save_name)
            print(f"  [Saved] {os.path.basename(save_name)}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"  Early stopping: {args.early_stop} epochs without improvement")
                break

    # ---- 测试最佳模型 ----
    best_files = sorted(glob.glob(args.save_path.replace('.pth', '_epoch*.pth')))
    if not best_files:
        print("警告: 未找到最佳模型文件")
        return
    best_file = sorted(best_files,
                       key=lambda x: float(x.split('_acc')[-1].replace('.pth', '')),
                       reverse=True)[0]
    print(f"\nDone. Best epoch {best_epoch}, val_acc {best_acc:.4f}")
    ckpt = torch.load(best_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print("  Loading best checkpoint for test evaluation ...")
    _, tm = validate(model, test_loader, criterion, device, desc="Test")
    print(f"Test  | Acc: {tm['accuracy']:.3f} | "
          f"PC: {tm['per_class'].get(0,0):.2f}/"
          f"{tm['per_class'].get(1,0):.2f}/"
          f"{tm['per_class'].get(2,0):.2f}")
    with open(args.save_path.replace(".pth", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {args.save_path.replace('.pth', '_history.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--lambda_contrast", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--cobre_dir", type=str, default=r"D:\Datasets\COBRE",
                        help="COBRE 数据集目录")
    parser.add_argument("--adhd_dir", type=str, default=r"D:\Datasets\ADHD-200",
                        help="ADHD-200 数据集目录")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)

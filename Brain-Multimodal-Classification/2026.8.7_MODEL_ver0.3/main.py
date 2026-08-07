"""main.py — 训练入口：数据加载、训练循环、CSV 记录、测试评估"""
import os, json, argparse, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models import BrainDiseaseModel, SMRIAugment
from losses import SupConLoss, mmd_loss
from trainer import (train_one_epoch, validate, compute_group_distances,
                     BalancedBatchSampler)
from domain import (DomainWrapper, CombatMorphWrapper, CombatFcWrapper)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr} | Latent: {args.latent_dim}")
    print(f"  λ_contrast: {args.lambda_contrast} | λ_ortho: {args.lambda_ortho} | λ_domain: {args.lambda_domain} | λ_mmd: {args.lambda_mmd} | λ_proto: {args.lambda_proto}")

    import sys, numpy as np
    sys.path.insert(0, r"D:\Datasets\COBRE")
    sys.path.insert(0, r"D:\Datasets\ADHD-200")
    from cobre_dataset import COBREDataset
    from adhd200_dataset import ADHD200Dataset

    print("[Data] Loading COBRE ...")
    cobre_ds = COBREDataset(split="all", seed=args.seed, normalize_smri=True, fisher_z=True,
                             base_dir=r"D:\Datasets\COBRE")
    print("[Data] Loading ADHD-200 ...")
    adhd_ds = ADHD200Dataset(split="all", seed=args.seed, normalize_smri=True, fisher_z=True,
                              base_dir=r"D:\Datasets\ADHD-200")

    # Label mapping: ADHD-200 1→2 (ADHD)
    adhd_ds.labels = adhd_ds.labels.copy()
    adhd_ds.labels[adhd_ds.labels == 1] = 2

    # 域标签：0=COBRE, 1=ADHD
    cobre_ds = DomainWrapper(cobre_ds, 0)
    adhd_ds  = DomainWrapper(adhd_ds, 1)

    # ComBat morph（可选）
    if args.combat_morph:
        cobre_combat = np.load(r"D:\Datasets\COBRE\morph_combat.npy")
        adhd_combat  = np.load(r"D:\Datasets\ADHD-200\morph_combat.npy")
        cobre_ds = CombatMorphWrapper(cobre_ds, cobre_combat)
        adhd_ds  = CombatMorphWrapper(adhd_ds,  adhd_combat)
        print("[ComBat] 已加载站点校正后的 morph 特征")

    # ComBat FC (可选)
    use_fc_input = False
    if args.combat_fc:
        cobre_fc = np.load(r"D:\Datasets\COBRE\fc_combat.npy")
        adhd_fc  = np.load(r"D:\Datasets\ADHD-200\fc_combat.npy")
        cobre_ds = CombatFcWrapper(cobre_ds, cobre_fc)
        adhd_ds  = CombatFcWrapper(adhd_ds,  adhd_fc)
        use_fc_input = True
        print("[ComBat] 已加载站点校正后的 FC 特征")

    from torch.utils.data import ConcatDataset
    combined_ds = ConcatDataset([cobre_ds, adhd_ds])
    all_labels = np.concatenate([cobre_ds.labels, adhd_ds.labels])
    print(f"[Data] 合并数据集: {len(combined_ds)} 样本 (COBRE: {len(cobre_ds)} + ADHD-200: {len(adhd_ds)})")
    print(f"  类别分布: 0={np.sum(all_labels==0)}, 1={np.sum(all_labels==1)}, 2={np.sum(all_labels==2)}")

    # Stratified split
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(all_labels):
        cls_idx = np.where(all_labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train:n_train+n_val])
        test_idx.append(cls_idx[n_train+n_val:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    test_idx = np.sort(np.concatenate(test_idx))
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 统一时间长度：所有 ROI 时序线性插值到 T_TARGET（此处扫描仅作诊断日志，确认数据范围）
    print("  Scanning max T ... ", end="", flush=True)
    global_max_T_bn = 0
    for i in range(len(cobre_ds)):
        t = cobre_ds.get_roi_T(i)
        global_max_T_bn = max(global_max_T_bn, t)
    for i in range(len(adhd_ds)):
        t = adhd_ds.get_roi_T(i)
        global_max_T_bn = max(global_max_T_bn, t)
    print(f"BN={global_max_T_bn}")

    # 时间插值目标长度（所有样本统一重采样到该长度）
    T_TARGET = args.t_target
    print(f"[Data] 时间插值: 所有 ROI TS 统一重采样到 T={T_TARGET}")

    def collate_fn(batch):
        out = {}
        for key in batch[0].keys():
            if key == "ROI_aal":
                continue  # AAL 图谱已移除，跳过
            vals = [b[key] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                # 可变长度 ROI 时序：先 per-ROI z-score（原长度），再线性插值统一到 T_TARGET
                if vals[0].ndim == 2 and key.startswith("ROI") and vals[0].shape[0] != vals[0].shape[1]:
                    # 先 per-ROI z-score（原长度），再插值到统一长度 150
                    resampled = []
                    for v in vals:
                        v = (v - v.mean(dim=1, keepdim=True)) / (v.std(dim=1, keepdim=True) + 1e-8)
                        if v.shape[1] != T_TARGET:
                            v = torch.nn.functional.interpolate(
                                v.unsqueeze(0), size=T_TARGET, mode="linear", align_corners=False).squeeze(0)
                        resampled.append(v)
                    out[key] = torch.stack(resampled)
                else:
                    out[key] = torch.stack(vals)
            elif key == "domain":
                out[key] = torch.tensor(vals, dtype=torch.long)
            elif key == "is_hc":
                out[key] = torch.tensor(vals, dtype=torch.bool)
        return out

    # 均衡采样训练集
    train_labels = all_labels[train_idx]
    batch_sampler = BalancedBatchSampler(train_labels, n_per_class=args.batch_size // 3)
    train_loader = DataLoader(Subset(combined_ds, train_idx),
                              batch_sampler=batch_sampler,
                              num_workers=0, collate_fn=collate_fn)
    print(f"  每 epoch {len(batch_sampler)} batches, 每类 {args.batch_size // 3} 样本")

    val_loader   = DataLoader(Subset(combined_ds, val_idx),   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(Subset(combined_ds, test_idx),  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = BrainDiseaseModel(num_classes=args.num_classes, latent_dim=args.latent_dim, dropout=args.dropout,
                              n_timepoints_bn=T_TARGET, use_fc_input=use_fc_input).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 域对抗
    domain_disc = None; domain_criterion = None
    if args.lambda_domain > 0:
        domain_disc = DomainDiscriminator(dim=args.latent_dim, num_classes=args.num_classes).to(device)
        domain_criterion = nn.CrossEntropyLoss()
        n_params += sum(p.numel() for p in domain_disc.parameters() if p.requires_grad)

    hc_domain_disc = None; hc_domain_criterion = None
    if args.lambda_hc_domain > 0:
        hc_domain_disc = HCDomainDiscriminator(in_dim=args.latent_dim * args.num_classes).to(device)
        hc_domain_criterion = nn.CrossEntropyLoss()
        n_params += sum(p.numel() for p in hc_domain_disc.parameters() if p.requires_grad)

    print(f"[Model] 可训练参数: {n_params:,} | 训练 batch: {len(train_loader)} | "
          f"验证 batch: {len(val_loader)} | 测试 batch: {len(test_loader)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_contrast = SupConLoss(temperature=0.07) if args.lambda_contrast > 0 else None
    params = list(model.parameters())
    if domain_disc is not None:
        params += list(domain_disc.parameters())
    if hc_domain_disc is not None:
        params += list(hc_domain_disc.parameters())
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "per_class": []}
    best_loss, best_epoch = float('inf'), 0
    patience_counter = 0
    total_t0 = time.time()

    # ── 输出文件夹 + CSV ──
    from datetime import datetime
    out_dir = os.path.join("OUTPUT", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "分类性能.csv")
    csv_cols = ["epoch", "train_loss", "val_loss", "accuracy",
                "macro_f1", "weighted_f1", "macro_auc",
                "precision_macro", "recall_macro",
                "recall_0", "recall_1", "recall_2",
                "f1_0", "f1_1", "f1_2"]
    with open(csv_path, "w") as f:
        f.write(",".join(csv_cols) + "\n")
    print(f"[CSV] {csv_path}")

    gen_csv_path = os.path.join(out_dir, "泛化能力.csv")
    gen_cols = ["epoch", "train_accuracy", "val_accuracy", "accuracy_gap",
                "train_loss", "val_loss", "loss_gap", "val_balanced_accuracy"]
    with open(gen_csv_path, "w") as f:
        f.write(",".join(gen_cols) + "\n")

    sep_csv_path = os.path.join(out_dir, "潜在空间可分性.csv")
    sep_cols = ["epoch", "latent_silhouette", "latent_inter_distance",
                "latent_intra_distance", "latent_separation_ratio"]
    with open(sep_csv_path, "w") as f:
        f.write(",".join(sep_cols) + "\n")

    mme_csv_path = os.path.join(out_dir, "多模态有效性.csv")
    mme_cols = ["epoch", "smri_acc", "fc_acc", "fusion_acc", "fusion_gain"]
    with open(mme_csv_path, "w") as f:
        f.write(",".join(mme_cols) + "\n")

    rel_csv_path = os.path.join(out_dir, "预测可依赖性.csv")
    rel_cols = ["epoch", "ece", "mean_confidence", "correct_confidence",
                "wrong_confidence", "confidence_gap"]
    with open(rel_csv_path, "w") as f:
        f.write(",".join(rel_cols) + "\n")

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        print(f"\n── Epoch {epoch}/{args.epochs} ──")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch,
                                      criterion_contrast=criterion_contrast,
                                      lambda_contrast=args.lambda_contrast,
                                      lambda_ortho=args.lambda_ortho,
                                      domain_disc=domain_disc,
                                      domain_criterion=domain_criterion,
                                      lambda_domain=args.lambda_domain,
                                      lambda_mmd=args.lambda_mmd,
                                      lambda_proto=args.lambda_proto,
                                      hc_domain_disc=hc_domain_disc,
                                      hc_domain_criterion=hc_domain_criterion,
                                      lambda_hc_domain=args.lambda_hc_domain)
        val_loss, m = validate(model, val_loader, criterion, device, desc="Val")
        sep = compute_group_distances(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(m["accuracy"])
        history["per_class"].append(m["per_class"])

        lr = scheduler.get_last_lr()[0]; scheduler.step()
        pc = m["per_class"]
        ep_time = time.time() - t_epoch
        total_time = time.time() - total_t0
        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {m['accuracy']:.3f} | "
              f"PC: {pc.get(0,0):.2f}/{pc.get(1,0):.2f}/{pc.get(2,0):.2f} | "
              f"LR: {lr:.2e} | {ep_time:.0f}s/epoch, total {total_time/60:.1f}min")

        # ── CSV 写入 ──
        with open(csv_path, "a") as f:
            row = [epoch, train_loss, val_loss, m["accuracy"],
                   m["macro_f1"], m["weighted_f1"], m["macro_auc"],
                   m["precision_macro"], m["recall_macro"],
                   m["recall_0"], m["recall_1"], m["recall_2"],
                   m["f1_0"], m["f1_1"], m["f1_2"]]
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")

        # ── 泛化能力 CSV ──
        bal_acc = (m["recall_0"] + m["recall_1"] + m["recall_2"]) / 3
        acc_gap = train_acc - m["accuracy"]
        loss_gap = val_loss - train_loss
        with open(gen_csv_path, "a") as f:
            row2 = [epoch, train_acc, m["accuracy"], acc_gap,
                    train_loss, val_loss, loss_gap, bal_acc]
            f.write(",".join(f"{v:.4f}" for v in row2) + "\n")

        # ── prototype 距离矩阵 ──
        if hasattr(model, 'classifier_proto'):
            with torch.no_grad():
                proto = torch.nn.functional.normalize(model.classifier_proto.prototypes, dim=-1)
                cos_mat = proto @ proto.T
                print(f"  Prototype cos: HC={cos_mat[0,0]:.3f}/{cos_mat[0,1]:.3f}/{cos_mat[0,2]:.3f}  "
                      f"SZ={cos_mat[1,0]:.3f}/{cos_mat[1,1]:.3f}/{cos_mat[1,2]:.3f}  "
                      f"ADHD={cos_mat[2,0]:.3f}/{cos_mat[2,1]:.3f}/{cos_mat[2,2]:.3f}")

            if hasattr(model.disease_fusion, 'token_bias'):
                with torch.no_grad():
                    b = next(iter(val_loader))
                    _, _, _, tk = model(b["sMRI"].to(device), b["sMRI_morph"].to(device),
                                        b["ROI_bn"].to(device))
                    tk_n = torch.nn.functional.normalize(tk, dim=-1)
                    p_n = torch.nn.functional.normalize(model.classifier_proto.prototypes, dim=-1)
                    cos_tp = (tk_n @ p_n.T).mean(0)
                    print(f"  Token->Proto: HC={cos_tp[0,0]:.3f}/{cos_tp[0,1]:.3f}/{cos_tp[0,2]:.3f}  "
                          f"SZ={cos_tp[1,0]:.3f}/{cos_tp[1,1]:.3f}/{cos_tp[1,2]:.3f}  "
                          f"ADHD={cos_tp[2,0]:.3f}/{cos_tp[2,1]:.3f}/{cos_tp[2,2]:.3f}")

        # ── 潜在空间可分性 CSV ──
        with open(sep_csv_path, "a") as f:
            row3 = [epoch, sep.get("latent_silhouette",0), sep.get("latent_inter_distance",0),
                    sep.get("latent_intra_distance",0), sep.get("latent_separation_ratio",0)]
            f.write(",".join(f"{v:.4f}" for v in row3) + "\n")

        # ── 多模态有效性 CSV ──
        with open(mme_csv_path, "a") as f:
            row4 = [epoch, sep.get("smri_acc",0), sep.get("fc_acc",0),
                    sep.get("fusion_acc",0), sep.get("fusion_gain",0)]
            f.write(",".join(f"{v:.4f}" for v in row4) + "\n")

        # ── 预测可依赖性 CSV ──
        with open(rel_csv_path, "a") as f:
            row5 = [epoch, m.get("ece",0), m.get("mean_confidence",0),
                    m.get("correct_confidence",0), m.get("wrong_confidence",0),
                    m.get("confidence_gap",0)]
            f.write(",".join(f"{v:.4f}" for v in row5) + "\n")

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            patience_counter = 0
            for old in glob.glob(os.path.join(out_dir, "best_model_epoch*.pth")):
                os.remove(old)
            save_name = os.path.join(out_dir,
                f'best_model_epoch{epoch}_train{train_loss:.4f}_val{val_loss:.4f}_acc{m["accuracy"]:.4f}.pth')
            torch.save(model.state_dict(), save_name)
            # 保存 prototype
            if hasattr(model, 'classifier_proto'):
                proto_dict = {"HC": model.classifier_proto.prototypes[0].cpu(),
                              "SZ": model.classifier_proto.prototypes[1].cpu(),
                              "ADHD": model.classifier_proto.prototypes[2].cpu()}
                torch.save(proto_dict, os.path.join(out_dir, "best_prototypes.pt"))
            print(f"  [Saved] {os.path.basename(save_name)}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"  Early stopping: {args.early_stop} epochs without improvement")
                break

    best_files = sorted(glob.glob(os.path.join(out_dir, "best_model_epoch*.pth")))
    if not best_files:
        print("警告: 未找到最佳模型文件")
        return
    best_file = sorted(best_files, key=lambda x: float(x.split('_val')[-1].split('_')[0]), reverse=False)[0]
    print(f"\nDone. Best epoch {best_epoch}, val_loss {best_loss:.4f}")
    ckpt = torch.load(best_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    print("  Loading best checkpoint for test evaluation ...")
    _, tm = validate(model, test_loader, criterion, device, desc="Test")
    print(f"Test  | Acc: {tm['accuracy']:.3f} | PC: {tm['per_class'].get(0,0):.2f}/{tm['per_class'].get(1,0):.2f}/{tm['per_class'].get(2,0):.2f}")
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {os.path.join(out_dir, 'history.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--t_target", type=int, default=150,
                        help="fMRI ROI 时间序列插值目标长度")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--lambda_contrast", type=float, default=0.1)
    parser.add_argument("--lambda_ortho", type=float, default=0.05,
                    help="正交性约束强度：0=关闭，0.01~0.1 推荐")
    parser.add_argument("--lambda_domain", type=float, default=0.0,
                    help="域对抗强度：0=关闭（推荐），0.05~0.2 实验性")
    parser.add_argument("--lambda_hc_domain", type=float, default=0.1,
                        help="HC-only 域对抗强度：0=关闭")
    parser.add_argument("--lambda_mmd", type=float, default=0.0,
                        help="MMD 域对齐强度：0=关闭")
    parser.add_argument("--lambda_proto", type=float, default=0.5,
                        help="Prototype loss 强度：0=关闭")
    parser.add_argument("--combat_morph", action="store_true",
                        help="使用 ComBat 站点校正后的 morph 特征")
    parser.add_argument("--combat_fc", action="store_true",
                    help="使用 ComBat 站点校正后的 FC 矩阵（替换 ROI TS）")
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    main(args)

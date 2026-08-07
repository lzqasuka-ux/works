"""trainer.py — 训练/验证/评估函数 + 均衡采样器"""
import time
import numpy as np
import torch
import torch.nn as nn

from models import SMRIAugment, BrainDiseaseModel
from losses import (SupConLoss, orthogonality_loss, mmd_loss,
                    consistency_loss, prototype_loss, token_diversity_loss)
from domain import (GradientReversalLayer, HCDomainDiscriminator,
                    DomainDiscriminator)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    criterion_contrast=None, lambda_contrast=0.1,
                    lambda_ortho=0.05,
                    lambda_proto=0.1,
                    domain_disc=None, domain_criterion=None, lambda_domain=0.1,
                    lambda_mmd=0.1,
                    hc_domain_disc=None, hc_domain_criterion=None, lambda_hc_domain=0.1):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_cont_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0
    t0 = time.time()
    n_batches = len(loader)
    use_contrast = criterion_contrast is not None and lambda_contrast > 0
    use_ortho = lambda_ortho > 0
    use_domain = domain_disc is not None and lambda_domain > 0
    use_mmd = lambda_mmd > 0
    use_hc_domain = hc_domain_disc is not None and lambda_hc_domain > 0
    use_proto = lambda_proto > 0
    total_ortho_loss = 0.0
    total_domain_loss = 0.0
    total_mmd_loss = 0.0
    total_hc_domain_loss = 0.0
    total_proto_loss = 0.0
    for i, batch in enumerate(loader):
        smri = batch["sMRI"].to(device)
        smri_morph = batch["sMRI_morph"].to(device)
        roi_bn = batch["ROI_bn"].to(device)
        labels = batch["label"].to(device)
        domains = batch.get("domain", None)
        is_hc = batch.get("is_hc", None)
        if use_domain and domains is not None:
            domains = domains.to(device)
        if use_hc_domain and is_hc is not None:
            is_hc = is_hc.to(device)

        if epoch == 1 and i == 0:
            print(f"  [Data] sMRI: {tuple(smri.shape)}, morph: {tuple(smri_morph.shape)}, "
                  f"ROI_bn: {tuple(roi_bn.shape)}")

        # Augmentation (仅 sMRI)
        smri_augmented = smri_aug(smri)
        # Instance norm：每个样本独立 z-score，消灭站点特征
        smri_augmented = (smri_augmented - smri_augmented.mean()) / (smri_augmented.std() + 1e-8)
        smri_morph = (smri_morph - smri_morph.mean(dim=-1, keepdim=True)) / (smri_morph.std(dim=-1, keepdim=True) + 1e-8)

        # ROI TS augmentation: mild noise + random masking (z-score 已在 collate 完成)
        roi_bn = roi_bn + torch.randn_like(roi_bn) * 0.005
        if torch.rand(1).item() < 0.3:
            mask_bn = (torch.rand_like(roi_bn) > 0.02).float()
            roi_bn = roi_bn * mask_bn

        optimizer.zero_grad()
        logits, z_disease, aux, tokens = model(smri_augmented, smri_morph, roi_bn)
        z_s, z_f = aux
        loss_cls = criterion(logits, labels)

        if use_contrast:
            loss_cont = criterion_contrast(z_disease, labels)
            loss = loss_cls + lambda_contrast * loss_cont
        else:
            loss_cont = torch.tensor(0.0, device=device)
            loss = loss_cls

        if use_ortho:
            loss_ortho = orthogonality_loss(z_s, z_f)
            loss = loss + lambda_ortho * loss_ortho
        else:
            loss_ortho = torch.tensor(0.0, device=device)

        if use_domain and domains is not None:
            loss_domain = domain_criterion(domain_disc(z_s, logits, alpha=min(0.1, epoch/10*0.1)), domains) \
                        + domain_criterion(domain_disc(z_f, logits, alpha=min(0.1, epoch/10*0.1)), domains)
            # 自适应：dom 越小（判别器越强）→ 权重越大 → 反向推力越大
            dom_scale = 1.0 / (loss_domain.item() + 0.1)
            loss = loss + lambda_domain * dom_scale * loss_domain
        else:
            loss_domain = torch.tensor(0.0, device=device)

        if use_mmd and domains is not None:
            loss_mmd = mmd_loss(z_s, domains, labels) + mmd_loss(z_f, domains, labels)
            loss_cons = consistency_loss(z_s, labels, domains) + consistency_loss(z_f, labels, domains)
            loss = loss + lambda_mmd * loss_mmd + lambda_mmd * 0.5 * loss_cons
        else:
            loss_mmd = torch.tensor(0.0, device=device)

        # ── HC-only 域对抗 ──
        if use_hc_domain and is_hc is not None and is_hc.sum() > 0:
            z_hc = z_disease[is_hc]
            d_hc = domains.to(device)[is_hc]
            alpha = min(0.1, epoch / 10 * 0.1)  # warmup
            loss_hc_domain = hc_domain_criterion(
                hc_domain_disc(z_hc, alpha=alpha), d_hc)
            with torch.no_grad():
                dom_pred = hc_domain_disc(z_hc, alpha=1.0).argmax(1)
                hc_dom_acc = (dom_pred == d_hc).float().mean().item()
            loss = loss + lambda_hc_domain * loss_hc_domain
        else:
            loss_hc_domain = torch.tensor(0.0, device=device)

        if use_proto:
            loss_proto = prototype_loss(tokens, labels, model.classifier_proto.prototypes)
            loss_div = token_diversity_loss(tokens)
            loss = loss + lambda_proto * (loss_proto + 0.3 * loss_div)
        else:
            loss_proto = torch.tensor(0.0, device=device)

        loss.backward()
        all_params = list(model.parameters())
        if domain_disc is not None:
            all_params += list(domain_disc.parameters())
        if hc_domain_disc is not None:
            all_params += list(hc_domain_disc.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_cont_loss += loss_cont.item() if use_contrast else 0.0
        total_train_correct += (logits.argmax(1) == labels).sum().item()
        total_train_samples += labels.size(0)
        total_ortho_loss += loss_ortho.item() if use_ortho else 0.0
        total_domain_loss += loss_domain.item() if use_domain else 0.0
        total_mmd_loss += loss_mmd.item() if use_mmd else 0.0
        total_hc_domain_loss += loss_hc_domain.item() if use_hc_domain else 0.0
        total_proto_loss += loss_proto.item() if use_proto else 0.0
        if (i+1) % 5 == 0 or i == n_batches - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (n_batches - i - 1)
            parts = [f"total={loss.item():.4f}", f"cls={loss_cls.item():.4f}"]
            if use_contrast:
                parts.append(f"cont={loss_cont.item():.4f}")
            if use_ortho:
                parts.append(f"ortho={loss_ortho.item():.4f}")
            if use_domain:
                parts.append(f"dom={loss_domain.item():.4f}")
            if use_mmd:
                parts.append(f"mmd={loss_mmd.item():.4f}")
            if use_hc_domain:
                parts.append(f"hc_dom={loss_hc_domain.item():.4f}(acc={hc_dom_acc:.2f})")
            if use_proto:
                parts.append(f"proto={loss_proto.item():.4f}")
            print(f"  Batch {i+1}/{n_batches} | Loss: {' '.join(parts)} | "
                  f"{elapsed:.0f}s elapsed, {eta:.0f}s remain", flush=True)
    avg = total_loss / n_batches
    parts_sum = [f"avg loss={avg:.4f}", f"cls={total_cls_loss/n_batches:.4f}"]
    if use_contrast:
        parts_sum.append(f"cont={total_cont_loss/n_batches:.4f}")
    if use_ortho:
        parts_sum.append(f"ortho={total_ortho_loss/n_batches:.4f}")
    if use_domain:
        parts_sum.append(f"dom={total_domain_loss/n_batches:.4f}")
    if use_mmd:
        parts_sum.append(f"mmd={total_mmd_loss/n_batches:.4f}")
    if use_hc_domain:
        parts_sum.append(f"hc_dom={total_hc_domain_loss/n_batches:.4f}")
    if use_proto:
        parts_sum.append(f"proto={total_proto_loss/n_batches:.4f}")
    print(f"  Training done: {time.time()-t0:.1f}s, {' '.join(parts_sum)}")

    train_acc = total_train_correct / max(total_train_samples, 1)

    # ── 梯度比例（仅第 1 epoch 第 1 batch）──
    if epoch == 1 and hasattr(model, 'classifier_proto'):
        print(f"  [Grad ratio] ", end="")
        try:
            b = next(iter(loader))
            smri_b = b["sMRI"].to(device)
            smri_b = (smri_b - smri_b.mean()) / (smri_b.std() + 1e-8)
            morph_b = b["sMRI_morph"].to(device)
            roi_b = b["ROI_bn"].to(device)
            lab_b = b["label"].to(device)
            domains_b = b.get("domain")
            if domains_b is not None: domains_b = domains_b.to(device)

            logits_b, zd_b, aux_b, tokens_b = model(smri_b, morph_b, roi_b)
            zs_b, zf_b = aux_b

            # CE only
            model.zero_grad()
            l_ce = criterion(logits_b, lab_b)
            l_ce.backward(retain_graph=True)
            g_ce = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            model.zero_grad()

            # SupCon only
            l_cont = criterion_contrast(zd_b, lab_b) if use_contrast else torch.tensor(0.0, device=device)
            if use_contrast:
                l_cont.backward(retain_graph=True)
                g_cont = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            else:
                g_cont = 0.0
            model.zero_grad()

            # Proto only
            l_proto = prototype_loss(tokens_b, lab_b, model.classifier_proto.prototypes) if use_proto else torch.tensor(0.0, device=device)
            if use_proto:
                l_proto.backward(retain_graph=True)
                g_proto = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            else:
                g_proto = 0.0
            model.zero_grad()

            g_sum = g_ce + g_cont + g_proto + 1e-8
            print(f"CE={g_ce/g_sum:.3f} SupCon={g_cont/g_sum:.3f} Proto={g_proto/g_sum:.3f}")
        except Exception as e:
            print(f"(skipped: {e})")

    return avg, train_acc

@torch.no_grad()


def validate(model, loader, criterion, device, desc="Val"):
    model.eval()
    total_loss, preds_list, labels_list, probs_list = 0.0, [], [], []
    n_batches = len(loader)
    print(f"  {desc} ... ", end="", flush=True)
    for i, batch in enumerate(loader):
        smri = batch["sMRI"].to(device)
        smri_morph = batch["sMRI_morph"].to(device)
        roi_bn = batch["ROI_bn"].to(device)
        labels = batch["label"].to(device)
        smri = (smri - smri.mean()) / (smri.std() + 1e-8)
        smri_morph = (smri_morph - smri_morph.mean(dim=-1, keepdim=True)) / (smri_morph.std(dim=-1, keepdim=True) + 1e-8)
        logits, _, _, _ = model(smri, smri_morph, roi_bn)
        total_loss += criterion(logits, labels).item()
        preds_list.append(torch.argmax(logits, 1).cpu())
        probs_list.append(torch.softmax(logits, 1).cpu())
        labels_list.append(labels.cpu())
    preds = torch.cat(preds_list)
    probs = torch.cat(probs_list)
    labs = torch.cat(labels_list)
    print("done", flush=True)

    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                                  roc_auc_score)
    acc = accuracy_score(labs, preds)
    macro_f1 = f1_score(labs, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labs, preds, average="weighted", zero_division=0)
    prec_macro = precision_score(labs, preds, average="macro", zero_division=0)
    rec_macro = recall_score(labs, preds, average="macro", zero_division=0)
    per_recall = recall_score(labs, preds, average=None, zero_division=0)
    per_f1 = f1_score(labs, preds, average=None, zero_division=0)
    try:
        auc = roc_auc_score(labs.numpy(), probs.numpy(), multi_class="ovr", average="macro")
    except:
        auc = 0.0

    # ── 预测可靠性 ──
    confs = probs.max(dim=1)[0].numpy()
    correct = (preds == labs).numpy()
    mean_conf = confs.mean()
    correct_conf = confs[correct].mean() if correct.sum() > 0 else 0.0
    wrong_conf = confs[~correct].mean() if (~correct).sum() > 0 else 0.0
    conf_gap = correct_conf - wrong_conf

    # ECE (10 bins)
    n_bins = 10
    ece = 0.0
    for b in range(n_bins):
        low, high = b / n_bins, (b + 1) / n_bins
        in_bin = (confs > low) & (confs <= high)
        if in_bin.sum() > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = confs[in_bin].mean()
            ece += (in_bin.sum() / len(confs)) * abs(acc_bin - conf_bin)

    per_class = {}
    for c in range(3):
        mask = (labs == c)
        per_class[c] = (preds[mask] == c).sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0

    metrics = {
        "accuracy": acc, "per_class": per_class,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1, "macro_auc": auc,
        "precision_macro": prec_macro, "recall_macro": rec_macro,
        "recall_0": per_recall[0] if len(per_recall)>0 else 0,
        "recall_1": per_recall[1] if len(per_recall)>1 else 0,
        "recall_2": per_recall[2] if len(per_recall)>2 else 0,
        "f1_0": per_f1[0] if len(per_f1)>0 else 0,
        "f1_1": per_f1[1] if len(per_f1)>1 else 0,
        "f1_2": per_f1[2] if len(per_f1)>2 else 0,
        # 预测可靠性
        "ece": ece,
        "mean_confidence": mean_conf,
        "correct_confidence": correct_conf,
        "wrong_confidence": wrong_conf,
        "confidence_gap": conf_gap,
    }
    return total_loss / n_batches, metrics


def compute_group_distances(model, loader, device):
    """打印类内/类间距离：同类该小，异类该大"""
    model.eval()
    accum = {}  # class → {"s": [], "f": [], "d": []}
    with torch.no_grad():
        for batch in loader:
            smri = batch["sMRI"].to(device)
            smri = (smri - smri.mean()) / (smri.std() + 1e-8)
            morph = batch["sMRI_morph"].to(device)
            roi = batch["ROI_bn"].to(device)
            labels = batch["label"].to(device)
            _, zd, (zs, zf), _ = model(smri, morph, roi)
            for i in range(len(labels)):
                c = labels[i].item()
                if c not in accum:
                    accum[c] = {"s": [], "f": [], "d": []}
                accum[c]["s"].append(zs[i])
                accum[c]["f"].append(zf[i])
                accum[c]["d"].append(zd[i])

    names = {0: "HC", 1: "SZ", 2: "ADHD"}
    for space, key in [("z_s", "s"), ("z_f", "f"), ("z_d", "d")]:
        # 类内：同类样本间平均 pairwise cosine 距离
        intra = []
        for c in sorted(accum):
            vecs = torch.stack(accum[c][key])
            vn = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
            cos_mat = vn @ vn.T
            mask = ~torch.eye(len(vecs), dtype=torch.bool, device=vecs.device)
            d = (1 - cos_mat[mask]).mean().item() if mask.sum() > 0 else 0.0
            intra.append(f"{names[c]}:{d:.3f}")
        # 类间：各类中心距离
        centers = {}
        for c in sorted(accum):
            vecs = torch.stack(accum[c][key])
            centers[c] = vecs.mean(dim=0)
            centers[c] = centers[c] / (centers[c].norm() + 1e-8)
        inter = []
        for i, ci in enumerate(sorted(centers)):
            for cj in sorted(centers)[i+1:]:
                d = (1 - (centers[ci] @ centers[cj]).item())
                inter.append(f"{names[ci]}-{names[cj]}:{d:.3f}")
        print(f"  [{key}] intra=[{' '.join(intra)}]  inter=[{' '.join(inter)}]")
    # Silhouette + DB on z_disease (余弦口径)
    sil = 0.0; db = -1.0
    try:
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        X = torch.cat([torch.stack(accum[c]["d"]) for c in sorted(accum)]).cpu().numpy()
        y = np.concatenate([[c]*len(accum[c]["d"]) for c in sorted(accum)])
        sil = silhouette_score(X, y, metric="cosine", random_state=42)
        db = davies_bouldin_score(X, y)
        print(f"  Silhouette={sil:.3f} (↑好)  DB={db:.3f} (↓好)")
    except Exception as e:
        print(f"  [silhouette 计算失败: {type(e).__name__}: {e}]")
        import traceback; traceback.print_exc()

    # ── 返回分离性指标（统一余弦距离口径）──
    sep = {"latent_silhouette": sil}
    try:
        all_centers = {}
        all_intra = []
        for c in sorted(accum):
            vecs = torch.stack(accum[c]["d"])                 # (n_c, D)
            center = vecs.mean(dim=0)                          # (D,)
            all_centers[c] = center
            # 类内余弦距离：样本到类中心的 1 - cos
            vn = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
            cn = center / (center.norm() + 1e-8)
            intra_c = (1 - (vn @ cn)).mean().item()
            all_intra.append(intra_c)
        inter_dists = []
        keys = sorted(all_centers)
        for i, ci in enumerate(keys):
            for cj in keys[i+1:]:
                # 类间余弦距离：两个类中心之间的 1 - cos
                ci_n = all_centers[ci] / (all_centers[ci].norm() + 1e-8)
                cj_n = all_centers[cj] / (all_centers[cj].norm() + 1e-8)
                inter_dists.append((1 - (ci_n @ cj_n)).item())
        sep["latent_inter_distance"] = np.mean(inter_dists) if inter_dists else 0.0
        sep["latent_intra_distance"] = np.mean(all_intra) if all_intra else 0.0
        sep["latent_separation_ratio"] = sep["latent_inter_distance"] / max(sep["latent_intra_distance"], 1e-8)
    except:
        sep = {"latent_silhouette": sil, "latent_inter_distance": 0.0,
               "latent_intra_distance": 0.0, "latent_separation_ratio": 0.0}

    # ── 多模态有效性 probe ──
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        probes = {}
        for key, name in [("s", "smri"), ("f", "fc"), ("d", "fusion")]:
            X = torch.cat([torch.stack(accum[c][key]) for c in sorted(accum)]).cpu().numpy()
            y = np.concatenate([[c]*len(accum[c][key]) for c in sorted(accum)])
            X = StandardScaler().fit_transform(X)
            clf = LogisticRegression(max_iter=500, random_state=42)
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            acc = cross_val_score(clf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=42)).mean()
            probes[name] = acc
        sep["smri_acc"] = probes["smri"]
        sep["fc_acc"] = probes["fc"]
        sep["fusion_acc"] = probes["fusion"]
        sep["fusion_gain"] = probes["fusion"] - max(probes["smri"], probes["fc"])
    except:
        sep["smri_acc"] = sep["fc_acc"] = sep["fusion_acc"] = sep["fusion_gain"] = 0.0

    return sep

# ---------------------------------------------------------------------------
# 均衡采样器
# ---------------------------------------------------------------------------


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


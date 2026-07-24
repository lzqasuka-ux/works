"""训练 & 验证工具函数"""

import time
import torch

from .augmentation import smri_aug


def compute_metrics(preds, labels, num_classes=3):
    """计算准确率与各类别准确率"""
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    acc = correct / total if total > 0 else 0.0
    per_class = {}
    for c in range(num_classes):
        mask = (labels == c)
        per_class[c] = (preds[mask] == c).sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0
    return {"accuracy": acc, "per_class": per_class}


def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    criterion_contrast=None, lambda_contrast=0.1):
    """单 epoch 训练"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_cont_loss = 0.0
    t0 = time.time()
    n_batches = len(loader)
    use_contrast = criterion_contrast is not None and lambda_contrast > 0

    for i, batch in enumerate(loader):
        smri = batch["sMRI"].to(device)
        smri_morph = batch["sMRI_morph"].to(device)
        roi_bn = batch["ROI_bn"].to(device)
        roi_aal = batch["ROI_aal"].to(device)
        labels = batch["label"].to(device)

        if epoch == 1 and i == 0:
            print(f"  [Data] sMRI: {tuple(smri.shape)}, morph: {tuple(smri_morph.shape)}, "
                  f"ROI_bn: {tuple(roi_bn.shape)}, ROI_aal: {tuple(roi_aal.shape)}")

        # Augmentation (仅 sMRI)
        smri_augmented = smri_aug(smri)

        # ROI TS augmentation: mild noise + random masking
        roi_bn = roi_bn + torch.randn_like(roi_bn) * 0.005
        roi_aal = roi_aal + torch.randn_like(roi_aal) * 0.005
        if torch.rand(1).item() < 0.3:
            mask_bn = (torch.rand_like(roi_bn) > 0.02).float()
            roi_bn = roi_bn * mask_bn
            mask_aal = (torch.rand_like(roi_aal) > 0.02).float()
            roi_aal = roi_aal * mask_aal

        optimizer.zero_grad()
        logits, z_disease, _ = model(smri_augmented, smri_morph, roi_bn, roi_aal)
        loss_cls = criterion(logits, labels)

        if use_contrast:
            loss_cont = criterion_contrast(z_disease, labels)
            loss = loss_cls + lambda_contrast * loss_cont
        else:
            loss_cont = torch.tensor(0.0, device=device)
            loss = loss_cls

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_cont_loss += loss_cont.item() if use_contrast else 0.0

        if (i + 1) % 5 == 0 or i == n_batches - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            parts = [f"total={loss.item():.4f}", f"cls={loss_cls.item():.4f}"]
            if use_contrast:
                parts.append(f"cont={loss_cont.item():.4f}")
            print(f"  Batch {i+1}/{n_batches} | Loss: {' '.join(parts)} | "
                  f"{elapsed:.0f}s elapsed, {eta:.0f}s remain", flush=True)

    avg = total_loss / n_batches
    parts_sum = [f"avg loss={avg:.4f}", f"cls={total_cls_loss/n_batches:.4f}"]
    if use_contrast:
        parts_sum.append(f"cont={total_cont_loss/n_batches:.4f}")
    print(f"  Training done: {time.time()-t0:.1f}s, {' '.join(parts_sum)}")
    return avg


@torch.no_grad()
def validate(model, loader, criterion, device, desc="Val"):
    """验证 / 测试"""
    model.eval()
    total_loss, preds_list, labels_list = 0.0, [], []
    n_batches = len(loader)
    print(f"  {desc} ... ", end="", flush=True)
    for i, batch in enumerate(loader):
        smri = batch["sMRI"].to(device)
        smri_morph = batch["sMRI_morph"].to(device)
        roi_bn = batch["ROI_bn"].to(device)
        roi_aal = batch["ROI_aal"].to(device)
        labels = batch["label"].to(device)
        logits, _, _ = model(smri, smri_morph, roi_bn, roi_aal)
        total_loss += criterion(logits, labels).item()
        preds_list.append(torch.argmax(logits, 1).cpu())
        labels_list.append(labels.cpu())
    preds = torch.cat(preds_list)
    labs = torch.cat(labels_list)
    print("done", flush=True)
    return total_loss / n_batches, compute_metrics(preds, labs)

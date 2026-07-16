"""Training utilities: train_one_epoch, validate, compute_metrics"""
import torch
import torch.nn as nn


def compute_metrics(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    acc = correct / total if total > 0 else 0.0
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return {
        "accuracy": acc,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(loader):
        smri = batch["sMRI"].to(device)
        fc = batch["FC"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits, _ = model(smri, fc)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 5 == 0 or i == len(loader) - 1:
            print(f"  Batch {i+1}/{len(loader)} Loss: {loss.item():.4f}", flush=True)
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_list, labels_list = 0.0, [], []
    for batch in loader:
        smri = batch["sMRI"].to(device)
        fc = batch["FC"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(smri, fc)
        total_loss += criterion(logits, labels).item()
        preds_list.append(torch.argmax(logits, 1).cpu())
        labels_list.append(labels.cpu())
    return total_loss / len(loader), compute_metrics(torch.cat(preds_list), torch.cat(labels_list))
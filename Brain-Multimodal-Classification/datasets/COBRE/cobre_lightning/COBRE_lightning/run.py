"""
run.py — 训练入口
==================
用法:
    python -m cobre_lightning.run --epochs 100 --batch_size 8 --lr 0.001 --latent_dim 32
"""
import os, json, argparse
import torch
import torch.nn as nn
import torch.optim as optim

# 数据加载（唯一的外部依赖）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cobre_dataset import get_dataloader

from .model import BrainDiseaseModel
from .trainer import train_one_epoch, validate


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Epochs: {args.epochs} | BS: {args.batch_size} | LR: {args.lr}")

    train_loader, val_loader, test_loader = get_dataloader(batch_size=args.batch_size, seed=args.seed)

    model = BrainDiseaseModel(num_classes=args.num_classes, latent_dim=args.latent_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_sensitivity": [], "val_specificity": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, m = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(m["accuracy"])
        history["val_sensitivity"].append(m["sensitivity"])
        history["val_specificity"].append(m["specificity"])

        lr = scheduler.get_last_lr()[0]; scheduler.step()
        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {m['accuracy']:.3f} | "
              f"Sens: {m['sensitivity']:.3f} | Spec: {m['specificity']:.3f} | LR: {lr:.2e}")

        if m["accuracy"] > best_acc:
            best_acc, best_epoch = m["accuracy"], epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc": best_acc}, args.save_path)
            print(f"  [Saved] Best val_acc = {best_acc:.4f}")

    print(f"\nDone. Best epoch {best_epoch}, val_acc {best_acc:.4f}")

    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, tm = validate(model, test_loader, criterion, device)
    print(f"Test  | Acc: {tm['accuracy']:.3f} | Sens: {tm['sensitivity']:.3f} | Spec: {tm['specificity']:.3f}")

    with open(args.save_path.replace(".pth", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="best_model_final.pth")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    main(args)
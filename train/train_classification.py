# train/train_classification.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.classification import ClassificationModel
from datasets.classification_dataset import build_classification_datasets
from utils.train_utils import train_one_epoch, evaluate
from torch.cuda.amp import GradScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--backbone', default='vit_base_patch16_224')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default='checkpoints/cls.pth')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds = build_classification_datasets(args.train_dir, args.val_dir, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=8, pin_memory=True)

    model = ClassificationModel(backbone_name=args.backbone, num_classes=args.num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler() if args.fp16 else None

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, args.out)
            print("Saved checkpoint")

if __name__ == "__main__":
    main()

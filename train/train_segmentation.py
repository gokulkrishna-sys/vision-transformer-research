# train/train_segmentation.py

import os
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.transforms import get_train_augmentations, get_val_augmentations
from models.segmentation import SimpleSegmentationModel
from utils.metrics import mean_iou

from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import numpy as np

# ---------- Simple VOC-style Dataset wrapper ----------
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        assert len(self.images) == len(self.masks), "Mismatch between images and masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import cv2
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # ensure mask is LongTensor for CE loss
        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask


# ---------- Training / Evaluation ----------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, log_interval=50):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, (images, masks) in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        if i % log_interval == 0:
            pbar.set_description(f"loss: {running_loss/(i+1):.4f}")
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        total_iou += mean_iou(preds, masks, num_classes)
        count += 1
    return total_loss / count, total_iou / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["save_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg["save_dir"], "logs"))

    # --- Datasets ---
    train_tfms = get_train_augmentations(cfg["img_size"])
    val_tfms = get_val_augmentations(cfg["img_size"])

    train_ds = SegmentationDataset(cfg["train_images"], cfg["train_masks"], transforms=train_tfms)
    val_ds = SegmentationDataset(cfg["val_images"], cfg["val_masks"], transforms=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=8, pin_memory=True)

    # --- Model ---
    model = SimpleSegmentationModel(
        encoder_name=cfg["encoder_name"],
        pretrained=cfg["pretrained"],
        encoder_out_index=cfg["encoder_out_index"],
        num_classes=cfg["num_classes"]
    ).to(device)

    # --- Loss, Optimizer ---
    criterion = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("fp16", False))

    # --- Training loop ---
    best_miou = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_miou = evaluate(model, val_loader, criterion, device, cfg["num_classes"])

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("mIoU/val", val_miou, epoch)

        # Save checkpoint
        if val_miou > best_miou:
            best_miou = val_miou
            ckpt_path = os.path.join(cfg["save_dir"], "best_model.pth")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "mIoU": val_miou}, ckpt_path)
            print(f"Saved new best checkpoint ({val_miou:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()

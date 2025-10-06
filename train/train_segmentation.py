# train/train_segmentation.py
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.segmentation_dataset import SegmentationDataset
from models.segmentation import build_segmentation_model

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_segmentation(cfg):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = SegmentationDataset(cfg["train_images"], cfg["train_masks"], cfg["img_size"])
    val_dataset = SegmentationDataset(cfg["val_images"], cfg["val_masks"], cfg["img_size"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    model = build_segmentation_model(cfg["timm_model"], num_classes=cfg["num_classes"], pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}")

        # Simple validation loop (IoU placeholder)
        model.eval()
        with torch.no_grad():
            total_iou = 0
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs).argmax(1)
                intersection = (preds & masks).float().sum((1,2))
                union = (preds | masks).float().sum((1,2))
                iou = (intersection / (union + 1e-6)).mean()
                total_iou += iou.item()
            print(f"Val IoU: {total_iou/len(val_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_segmentation(cfg)

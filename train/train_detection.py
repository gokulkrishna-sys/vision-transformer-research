# train/train_detection.py
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datasets.detection_dataset import DetectionDataset
from models.detection import build_faster_rcnn
import torch.optim as optim
from tqdm import tqdm

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train_detection(cfg):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = DetectionDataset(cfg["train_images"], cfg["train_annotations"])
    val_dataset = DetectionDataset(cfg["val_images"], cfg["val_annotations"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = build_faster_rcnn(timm_name=cfg["timm_model"], num_classes=cfg["num_classes"], pretrained_backbone=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        lr_scheduler.step()
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_detection(cfg)

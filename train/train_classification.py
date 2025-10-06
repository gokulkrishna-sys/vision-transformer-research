# train/train_classification.py
import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataloaders(cfg):
    transform_train = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(cfg["train_dir"], transform=transform_train)
    val_dataset = datasets.ImageFolder(cfg["val_dir"], transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main(cfg):
    os.makedirs(cfg["save_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg["save_dir"], "runs"))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(cfg)

    model = create_model(cfg["backbone"], pretrained=cfg["pretrained"], num_classes=cfg["num_classes"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_acc = 0.0
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

        print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc and cfg["save_best"]:
            torch.save(model.state_dict(), os.path.join(cfg["save_dir"], "best_model.pth"))
            best_acc = val_acc

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)

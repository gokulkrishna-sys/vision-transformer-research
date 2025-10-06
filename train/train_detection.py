# train/train_detection.py
import torch
from torch.utils.data import DataLoader
from models.detection import build_faster_rcnn
from datasets.detection_dataset import DetectionDataset  
from tqdm import tqdm
import torch.optim as optim

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)

def train_detection(train_dataset, val_dataset, timm_name='resnet50', num_classes=91, device='cuda', epochs=12, batch=4):
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=8)

    model = build_faster_rcnn(timm_name=timm_name, num_classes=num_classes, pretrained_backbone=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f"Epoch {epoch} done")
        # Evaluation: run model in eval mode, accumulate mAP via COCO API or custom metric

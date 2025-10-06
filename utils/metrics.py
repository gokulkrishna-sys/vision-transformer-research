# utils/metrics.py
import torch

def mean_iou(preds, targets, num_classes):
    """
    Compute mean IoU for semantic segmentation.
    preds, targets: (B,H,W) LongTensors
    """
    ious = []
    preds = preds.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)

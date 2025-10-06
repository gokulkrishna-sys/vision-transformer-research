# models/segmentation.py
import torch.nn as nn
import torch
from collections import OrderedDict
from models.backbones import create_timm_features_backbone
import torch.nn.functional as F

class SimpleSegmentationModel(nn.Module):
    def __init__(self, encoder_name='mit_b2', pretrained=True, encoder_out_index=-1, num_classes=21):
        super().__init__()
        # encoder returns list -> take last feature as bottleneck
        self.encoder = create_timm_features_backbone(name=encoder_name, pretrained=pretrained, out_indices=(encoder_out_index,))
        # try to get channels
        channels = self.encoder.model.feature_info.channels() if hasattr(self.encoder.model, "feature_info") else None
        if channels is None:
            raise RuntimeError("Encoder must expose channels via feature_info")
        in_channels = channels[encoder_out_index]
        self.decode_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feats = self.encoder(x)
        # feats is OrderedDict; take last
        last_feat = list(feats.values())[-1]
        logits = self.decode_head(last_feat)
        # upsample to input size
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

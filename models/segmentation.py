# models/segmentation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

class SegmentationHead(nn.Module):
    """
    Simple decoder head for segmentation.
    Uses a few ConvTranspose2d layers to upsample feature maps.
    """
    def __init__(self, in_channels, num_classes, decoder_channels=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, decoder_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(decoder_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)


class ViTSegmentationModel(nn.Module):
    """
    Segmentation model with a timm backbone and a simple decoder head.
    Works for both CNN and ViT backbones.
    """
    def __init__(self, backbone_name="vit_base_patch16_224", num_classes=21, pretrained=True):
        super().__init__()
        # Load backbone (timm model)
        self.encoder = create_model(backbone_name, features_only=True, pretrained=pretrained)
        encoder_channels = self.encoder.feature_info[-1]['num_chs']

        # Decoder / Segmentation head
        self.seg_head = SegmentationHead(encoder_channels, num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        x = feats[-1]  # Use final feature map
        x = self.seg_head(x)
        # Resize logits to match input image size
        x = F.interpolate(x, size=(feats[0].shape[2]*16, feats[0].shape[3]*16), mode="bilinear", align_corners=False)
        return x


def build_segmentation_model(timm_name="vit_base_patch16_224", num_classes=21, pretrained=True):
    """
    Factory function to build a segmentation model.
    """
    return ViTSegmentationModel(backbone_name=timm_name, num_classes=num_classes, pretrained=pretrained)

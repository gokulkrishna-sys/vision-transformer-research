# models/classification.py
import torch
import torch.nn as nn
import timm

class ClassificationModel(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_224', num_classes=1000, pretrained=True, pool='avg', dropout=0.0):
        super().__init__()
        # Use timm's model but disable its classifier, we'll use feature extraction
        model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool=pool)
        # The model now returns a feature vector (embedding). We'll add our head.
        embed_dim = model.num_features
        self.backbone = model
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim, num_classes)
        )
    def forward(self, x):
        x = self.backbone.forward_features(x)  # timm models provide .forward_features()
        x = self.backbone.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.head(x)
        return logits

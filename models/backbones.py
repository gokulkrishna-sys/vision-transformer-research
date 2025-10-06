# models/backbones.py
import timm
import torch.nn as nn
from collections import OrderedDict

def create_timm_features_backbone(name='resnet50', pretrained=True, features_only=True, out_indices=None):
    """
    Returns a feature-extractor model from timm that outputs intermediate feature maps as an OrderedDict.
    Use features_only=True and specify out_indices to pick layers.
    """
    # Many timm models support features_only
    model = timm.create_model(name, pretrained=pretrained, features_only=features_only, out_indices=out_indices)

    # model.forward returns list of feature maps; wrap to return OrderedDict with names
    class FeatureBackbone(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # channel counts: available at model.feature_info.channels() if supported
            try:
                self.channels = model.feature_info.channels()
            except Exception:
                # fallback - user must know channel sizes
                self.channels = None

        def forward(self, x):
            feats = self.model(x)  # list/tuple
            if isinstance(feats, (list, tuple)):
                names = [f"feat{i+1}" for i in range(len(feats))]
                return OrderedDict([(n, f) for n, f in zip(names, feats)])
            else:
                return OrderedDict([("feat1", feats)])

    return FeatureBackbone(model)

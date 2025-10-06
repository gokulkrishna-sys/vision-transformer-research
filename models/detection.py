# models/detection.py
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from models.backbones import create_timm_features_backbone
import torch

def timm_to_fpn_backbone(timm_name='resnet50', pretrained=True, out_indices=(1,2,3), fpn_out_channels=256):
    """
    Create a torchvision-compatible backbone with FPN using timm feature extractor.
    out_indices refer to timm feature outputs indices to use for FPN.
    """
    # Create timm feature extractor; timm feature returns a list of tensors
    feature_extractor = create_timm_features_backbone(name=timm_name, pretrained=pretrained, out_indices=out_indices)

    # we need channel counts
    if hasattr(feature_extractor.model, "feature_info"):
        in_channels_list = feature_extractor.model.feature_info.channels()[out_indices[0]:out_indices[-1]+1] \
            if isinstance(out_indices, (list,tuple)) else feature_extractor.model.feature_info.channels()
    else:
        # fallback - user should set channels manually
        raise RuntimeError("timm backbone does not expose channel sizes. Use a timm model with features_only=True supporting feature_info.")

    # wrap into a nn.Module that returns OrderedDict with keys that BackboneWithFPN expects
    class Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.body = backbone
        def forward(self, x):
            od = self.body(x)
            return od

    body = Wrapper(feature_extractor)
    backbone_with_fpn = BackboneWithFPN(body, in_channels_list=in_channels_list, out_channels=fpn_out_channels)
    return backbone_with_fpn

def build_faster_rcnn(timm_name='resnet50', num_classes=91, pretrained_backbone=True):
    backbone = timm_to_fpn_backbone(timm_name=timm_name, pretrained=pretrained_backbone)
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

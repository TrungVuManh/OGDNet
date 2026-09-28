"""
DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution
Chen et al., ECCV 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""

    def __init__(self, in_channels: int, out_channels: int, dilations: list = [6, 12, 18]):
        super().__init__()

        modules = []

        # 1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Atrous convolutions
        for dilation in dilations:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Global average pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs[:-1]:
            res.append(conv(x))

        # Global pooling branch
        global_feat = self.convs[-1](x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(global_feat)

        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        backbone: Backbone network name
        pretrained: Use pretrained backbone
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        backbone: str = 'resnet50',
        pretrained: bool = True
    ):
        super().__init__()

        # Backbone (using timm)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            output_stride=16
        )

        # Get feature channels
        backbone_channels = self.backbone.feature_info.channels()

        # ASPP
        self.aspp = ASPP(backbone_channels[-1], 256)

        # Low-level features processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(backbone_channels[1], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Classifier
        self.classifier = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        input_shape = x.shape[2:]

        # Backbone features
        features = self.backbone(x)
        low_level_feat = features[1]  # 1/4 resolution
        high_level_feat = features[-1]  # 1/16 resolution

        # ASPP
        aspp_out = self.aspp(high_level_feat)

        # Upsample ASPP output
        aspp_out = F.interpolate(aspp_out, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)

        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)

        # Concatenate
        concat = torch.cat([aspp_out, low_level_feat], dim=1)

        # Decoder
        decoder_out = self.decoder(concat)

        # Classifier
        out = self.classifier(decoder_out)

        # Upsample to input resolution
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)

        return torch.sigmoid(out)


def build_model(config: Dict) -> DeepLabV3Plus:
    model_config = config.get('model', config)
    return DeepLabV3Plus(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 1),
        backbone=model_config.get('backbone', 'resnet50'),
        pretrained=model_config.get('pretrained', True)
    )

"""
Swin Transformer + UPerNet decoder for semantic segmentation.
Encoder: Liu et al. "Swin Transformer." ICCV 2021.
Decoder: Xiao et al. "Unified Perceptual Parsing." ECCV 2018.

Architecture:
  Swin encoder (timm: swin_tiny_patch4_window7_224) → 4 multi-scale features
  UPerNet decoder (PPM + FPN + fusion head) → binary mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict


class PPM(nn.Module):
    """Pyramid Pooling Module applied to the deepest encoder feature."""

    def __init__(self, in_channels: int, pool_sizes: tuple = (1, 2, 3, 6)):
        super().__init__()
        inter = in_channels // len(pool_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_channels, inter, 1, bias=False),
                nn.BatchNorm2d(inter),
                nn.ReLU(inplace=True),
            )
            for s in pool_sizes
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + inter * len(pool_sizes), in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pooled = [F.interpolate(s(x), size=(h, w), mode='bilinear', align_corners=False)
                  for s in self.stages]
        return self.fuse(torch.cat([x] + pooled, dim=1))


class UPerNetDecoder(nn.Module):
    """UPerNet: PPM on deepest feature + FPN top-down fusion + head."""

    def __init__(self, encoder_channels: List[int], fpn_channels: int = 256, out_channels: int = 1):
        super().__init__()

        self.ppm = PPM(encoder_channels[-1])

        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for c in encoder_channels
        ])

        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in encoder_channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_channels * len(encoder_channels), fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(fpn_channels, out_channels, 1)

    def forward(self, features: List[torch.Tensor], output_size) -> torch.Tensor:
        feats = list(features)
        feats[-1] = self.ppm(feats[-1])

        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]

        # Top-down FPN
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='bilinear', align_corners=False,
            )

        # Smooth + upsample all to finest scale
        target = laterals[0].shape[-2:]
        fpn_outs = []
        for smooth, lat in zip(self.smooth, laterals):
            x = smooth(lat)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            fpn_outs.append(x)

        out = self.fuse(torch.cat(fpn_outs, dim=1))
        out = self.head(out)
        return F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)


class SwinUPerNet(nn.Module):
    """
    Swin Transformer encoder with UPerNet decoder.

    Args:
        encoder_name: timm Swin model name
        pretrained:   load ImageNet weights
        fpn_channels: channel dimension throughout the FPN
        out_channels: number of output segmentation channels
    """

    def __init__(
        self,
        encoder_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        fpn_channels: int = 256,
        out_channels: int = 1,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=512,
        )
        enc_chs = self.encoder.feature_info.channels()
        # swin_tiny: [96, 192, 384, 768] @ strides [4, 8, 16, 32]

        self.decoder = UPerNetDecoder(enc_chs, fpn_channels=fpn_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        # timm Swin returns NHWC [B,H,W,C] → convert to NCHW [B,C,H,W]
        features = [f.permute(0, 3, 1, 2).contiguous() for f in self.encoder(x)]
        out = self.decoder(features, input_size)
        return torch.sigmoid(out)


def build_model(config: Dict) -> SwinUPerNet:
    model_config = config.get('model', config)
    return SwinUPerNet(
        encoder_name=model_config.get('encoder_name', 'swin_tiny_patch4_window7_224'),
        pretrained=model_config.get('pretrained', True),
        fpn_channels=model_config.get('fpn_channels', 256),
        out_channels=model_config.get('out_channels', 1),
    )

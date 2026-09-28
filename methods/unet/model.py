"""
U-Net (Ronneberger et al., MICCAI 2015) with a pretrained backbone encoder.

Originally U-Net used an encoder trained from scratch. To match the other
methods in this benchmark (DeepLabV3+, ConvNeXt-UNet, OGDNet ...) the
contracting path is replaced by an ImageNet-pretrained timm backbone, while
the symmetric expanding path (DoubleConv decoder + skip connections) is kept.

Architecture:
  Encoder : timm backbone (resnet18/34/50/101, convnext_tiny ...), pretrained
  Decoder : N × (upsample + concat skip + DoubleConv)
  Head    : Conv2d(c, 1) → bilinear upsample to input size → Sigmoid

The decoder adapts automatically to the backbone's channel widths and to the
number of feature levels (5 for ResNet, 4 for ConvNeXt).
"""

from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsample (×2) → optional concat skip → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None and skip.shape[1] > 0:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net with a pretrained timm encoder.

    Backbone options (set model.backbone):
      'resnet18' / 'resnet34' → 5 levels, ch=[64, 64,128,256,512]
      'resnet50' / 'resnet101'→ 5 levels, ch=[64,256,512,1024,2048]
      'convnext_tiny'         → 4 levels, ch=[96,192,384,768]

    decoder_channels gives the per-stage output widths from deepest to shallowest.
    """

    def __init__(
        self,
        backbone: str               = 'resnet34',
        pretrained: bool            = True,
        decoder_channels: List[int] = None,
        in_channels: int            = 3,
        out_channels: int           = 1,
    ):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        n_levels = 4 if backbone.startswith('convnext') else 5
        out_idx  = tuple(range(n_levels))

        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            features_only=True, out_indices=out_idx, in_chans=in_channels,
        )
        enc_ch = list(self.encoder.feature_info.channels())

        dc = decoder_channels
        if n_levels == 5:
            self.dec4 = UpBlock(enc_ch[4], enc_ch[3], dc[0])
            self.dec3 = UpBlock(dc[0],     enc_ch[2], dc[1])
            self.dec2 = UpBlock(dc[1],     enc_ch[1], dc[2])
            self.dec1 = UpBlock(dc[2],     enc_ch[0], dc[3])
            self.dec0 = UpBlock(dc[3],     0,         dc[4])
            out_ch = dc[4]
            self._n_stages = 5
        else:
            self.dec3 = UpBlock(enc_ch[3], enc_ch[2], dc[0])
            self.dec2 = UpBlock(dc[0],     enc_ch[1], dc[1])
            self.dec1 = UpBlock(dc[1],     enc_ch[0], dc[2])
            self.dec0 = UpBlock(dc[2],     0,         dc[3])
            out_ch = dc[3]
            self._n_stages = 4

        self.head = nn.Sequential(
            nn.Conv2d(out_ch, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feats = self.encoder(x)

        # Defensive NHWC → NCHW (some timm backbones output channels-last)
        enc_ch = list(self.encoder.feature_info.channels())
        feats = [f.permute(0, 3, 1, 2).contiguous()
                 if f.shape[1] != c and f.shape[-1] == c else f
                 for f, c in zip(feats, enc_ch)]

        if self._n_stages == 5:
            f0, f1, f2, f3, f4 = feats
            d = self.dec4(f4, f3)
            d = self.dec3(d,  f2)
            d = self.dec2(d,  f1)
            d = self.dec1(d,  f0)
            d = self.dec0(d)
        else:
            f0, f1, f2, f3 = feats
            d = self.dec3(f3, f2)
            d = self.dec2(d,  f1)
            d = self.dec1(d,  f0)
            d = self.dec0(d)

        out = self.head(d)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        return out


def build_model(config: Dict) -> UNet:
    cfg = config.get('model', config)
    return UNet(
        backbone         = cfg.get('backbone', 'resnet34'),
        pretrained       = cfg.get('pretrained', True),
        decoder_channels = cfg.get('decoder_channels', [256, 128, 64, 32, 16]),
        in_channels      = cfg.get('in_channels', 3),
        out_channels     = cfg.get('out_channels', 1),
    )

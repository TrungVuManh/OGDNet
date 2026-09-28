"""
ResNet + FPN decoder for crack segmentation.
He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.

Originally the ResNet encoder here was trained from scratch. To match the other
methods in this benchmark it now uses an ImageNet-pretrained timm backbone, with
the same FPN top-down decoder on top.

Architecture:
  Encoder : timm backbone (resnet18/34/50/101 ...), pretrained
            4 feature maps at strides [4, 8, 16, 32]
  Neck    : FPN top-down pathway (fpn_ch at each level)
  Head    : Conv(fpn_ch,128) → Conv(128,1) → bilinear upsample → Sigmoid
"""

from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ResNetFPN(nn.Module):
    """
    Pretrained ResNet encoder + FPN decoder for binary crack segmentation.

    Backbone options (set model.backbone):
      'resnet18' / 'resnet34' → FPN over ch=[64,128,256,512]
      'resnet50' / 'resnet101'→ FPN over ch=[256,512,1024,2048]

    fpn_ch : channels at each FPN level.
    """

    def __init__(
        self,
        backbone: str    = 'resnet50',
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 1,
        fpn_ch: int      = 256,
    ):
        super().__init__()

        # Encoder — keep all reduction levels, then use the 4 highest strides
        # (strides 4, 8, 16, 32). For ResNet timm this drops the stride-2 stem map.
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            features_only=True, in_chans=in_channels,
        )
        enc_ch = list(self.encoder.feature_info.channels())[-4:]   # C1..C4

        # FPN laterals (1×1) — one per level, ordered shallow → deep
        self.laterals = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in enc_ch])
        # 3×3 smoothing after each top-down merge (all but the deepest level)
        self.smooths = nn.ModuleList(
            [nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1) for _ in range(len(enc_ch) - 1)]
        )

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(fpn_ch, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 1),
        )

    @staticmethod
    def _upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=True) + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        feats = self.encoder(x)[-4:]   # [C1, C2, C3, C4], shallow → deep
        enc_ch = list(self.encoder.feature_info.channels())[-4:]
        feats = [f.permute(0, 3, 1, 2).contiguous()
                 if f.shape[1] != c and f.shape[-1] == c else f
                 for f, c in zip(feats, enc_ch)]

        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]

        # Top-down pathway: start from deepest, merge upward
        p = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            p = self.smooths[i](self._upsample_add(p, laterals[i]))

        out = self.head(p)                       # finest FPN level (stride 4)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        return torch.sigmoid(out)


def build_model(config: Dict) -> ResNetFPN:
    cfg = config.get('model', config)
    return ResNetFPN(
        backbone     = cfg.get('backbone', 'resnet50'),
        pretrained   = cfg.get('pretrained', True),
        in_channels  = cfg.get('in_channels', 3),
        out_channels = cfg.get('out_channels', 1),
        fpn_ch       = cfg.get('fpn_ch', 256),
    )

"""
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.
Xie et al., NeurIPS 2021.

Encoder : Mix Transformer (MiT) via timm — pretrained, 4 feature levels
Decoder : All-MLP head — lightweight, no convolutions in projection
"""

from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError('timm is required. Install with: pip install timm')


class SegFormerHead(nn.Module):
    """
    All-MLP decoder head.
    Projects each encoder stage to embed_dim, upsamples to stride-4,
    concatenates and fuses, then predicts.
    """

    def __init__(self, in_channels: List[int], embed_dim: int = 256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(c, embed_dim) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.head    = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        H, W = features[0].shape[-2:]   # stride-4 resolution

        projected = []
        for feat, linear in zip(features, self.proj):
            B, C, h, w = feat.shape
            x = feat.flatten(2).transpose(1, 2)   # [B, h*w, C]
            x = linear(x)                          # [B, h*w, embed_dim]
            x = x.transpose(1, 2).reshape(B, -1, h, w)
            if (h, w) != (H, W):
                x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            projected.append(x)

        x = self.fuse(torch.cat(projected, dim=1))
        x = self.dropout(x)
        return self.head(x)


class SegFormer(nn.Module):
    def __init__(
        self,
        backbone: str  = 'mit_b2',
        embed_dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        enc_ch = self.encoder.feature_info.channels()
        self.decode_head = SegFormerHead(enc_ch, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W   = x.shape[-2:]
        feats  = self.encoder(x)
        out    = self.decode_head(feats)
        out    = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return self.sigmoid(out)


def build_model(config: Dict) -> SegFormer:
    cfg = config.get('model', config)
    return SegFormer(
        backbone  = cfg.get('backbone',  'mit_b2'),
        embed_dim = cfg.get('embed_dim', 256),
        pretrained= cfg.get('pretrained', True),
    )

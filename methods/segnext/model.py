"""
SegNeXt: Rethinking Convolutional Attention Design in Semantic Segmentation.
Guo et al., NeurIPS 2022.

Architecture:
  MSCAN-B encoder (custom implementation, 4 stages)
  All-MLP segmentation head → binary mask

MSCAN-B config: channels=[64,128,320,512], depths=[3,3,12,3]
"""

from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MSCAN building blocks
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dw(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1   = nn.Linear(in_features, hidden_features)
        self.dw    = DWConv(hidden_features)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden_features, in_features)
        self.drop  = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dw(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MSCAAttention(nn.Module):
    """Multi-Scale Convolutional Attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv0  = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1  = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv1_ = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv2  = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv2_ = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv3  = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv3_ = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv_sq = nn.Conv2d(dim, dim, 1)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u   = x.clone()
        a   = self.conv0(x)
        a1  = self.conv1_(self.conv1(a))
        a2  = self.conv2_(self.conv2(a))
        a3  = self.conv3_(self.conv3(a))
        a   = self.conv_sq(a + a1 + a2 + a3)
        return u * a


class MSCABlock(nn.Module):
    """One MSCAN transformer block."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.attn   = MSCAAttention(dim)
        self.mlp    = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        # Attention branch
        residual = x
        x = self.norm1(x).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.attn(x)
        x = x.flatten(2).transpose(1, 2) + residual
        # MLP branch
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, patch_size: int, stride: int):
        super().__init__()
        pad = patch_size // 2
        self.proj = nn.Conv2d(in_ch, out_ch, patch_size, stride=stride, padding=pad)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# MSCAN encoder
# ---------------------------------------------------------------------------

_CONFIGS = {
    # (channels, depths)
    't': ([32,  64,  160, 256], [3, 3,  5, 2]),
    's': ([64,  128, 320, 512], [2, 2,  4, 2]),
    'b': ([64,  128, 320, 512], [3, 3, 12, 3]),
    'l': ([64,  128, 320, 512], [3, 5, 27, 3]),
}


class MSCAN(nn.Module):
    """MSCAN backbone (4 stages, features_only-style output)."""

    def __init__(self, variant: str = 'b', in_channels: int = 3):
        super().__init__()
        channels, depths = _CONFIGS[variant]

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_channels if i == 0 else channels[i - 1],
                              channels[i],
                              patch_size=7 if i == 0 else 3,
                              stride=4 if i == 0 else 2)
            for i in range(4)
        ])

        self.stages = nn.ModuleList([
            nn.ModuleList([MSCABlock(channels[i]) for _ in range(depths[i])])
            for i in range(4)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(channels[i]) for i in range(4)])
        self._channels = channels

    @property
    def feature_info(self):
        class _FI:
            def channels(self_inner):
                return self._channels
        return _FI()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)   # x: [B, H*W, C]
            for blk in self.stages[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            B, _, C = x.shape
            feat = x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            outs.append(feat)
            x = feat  # spatial tensor for next OverlapPatchEmbed
        return outs


# ---------------------------------------------------------------------------
# All-MLP head + full model
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim: int = 256):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, features: List[torch.Tensor], output_size) -> torch.Tensor:
        target = features[0].shape[-2:]
        projected = []
        for proj, f in zip(self.projects, features):
            x = proj(f)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            projected.append(x)
        x = self.fuse(torch.cat(projected, dim=1))
        x = self.head(x)
        return F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)


class SegNeXt(nn.Module):
    def __init__(self, variant: str = 'b', embed_dim: int = 256):
        super().__init__()
        self.encoder = MSCAN(variant)
        enc_chs = self.encoder.feature_info.channels()
        self.head = MLPHead(enc_chs, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        out = self.head(features, x.shape[-2:])
        return torch.sigmoid(out)


def build_model(config: Dict) -> SegNeXt:
    cfg = config.get('model', config)
    variant = cfg.get('variant', 'b')
    # legacy: if encoder_name like 'mscan_b' was set, parse the variant
    enc = cfg.get('encoder_name', '')
    if enc.startswith('mscan_'):
        variant = enc.split('_')[-1]
    return SegNeXt(
        variant=variant,
        embed_dim=cfg.get('embed_dim', 256),
    )

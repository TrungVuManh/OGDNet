"""
OGDNet: Orientation-Guided Deformable Network for crack segmentation.

Architecture:
  Encoder  : timm backbone (resnet18/resnet34/resnet50/resnet101/convnext_tiny, pretrained)
  Orient   : OrientationHead on f_mid (stride-16) â†’ Î¸ map
  Decoder  : 4 Ã— OGDDecoder stages + 1 final upsample
  Head     : 1Ã—1 conv â†’ sigmoid

Key stability fixes:
  - offset_refine last layer is ZERO-initialized â†’ model starts as pure
    rotation-only deformable conv; residual branch grows only as needed
  - offsets are soft-clamped via tanh (no unbounded gradients)
  - base_scale learned parameter starts at 0 â†’ identity; ramps up smoothly
"""

from __future__ import annotations
from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import timm


# ---------------------------------------------------------------------------
# Orientation Head
# ---------------------------------------------------------------------------

class OrientationHead(nn.Module):
    """
    Predicts crack orientation Î¸ âˆˆ (-Ï€/2, Ï€/2] using Ï€-periodic encoding.

    Predicts (cos2Î¸, sin2Î¸) so there is no discontinuity at Â±Ï€/2.
    Î¸ = atan2(sin2Î¸, cos2Î¸) / 2
    """

    def __init__(self, in_channels: int, mid_channels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, 1),
        )
        # Small init on output layer â†’ (cos2Î¸, sin2Î¸) are small but nonzero
        nn.init.normal_(self.conv[-1].weight, std=0.01)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cs   = self.conv(x)
        # L2-normalize to unit circle before atan2 â€” prevents (0,0) â†’ NaN gradient
        cs   = F.normalize(cs, dim=1, eps=1e-8)
        return torch.atan2(cs[:, 1:2], cs[:, 0:1]) / 2.0   # [B, 1, H, W]


# ---------------------------------------------------------------------------
# OGDConvBlock
# ---------------------------------------------------------------------------

DEFAULT_ABLATION = {
    'use_orientation': True,
    'use_deformable': True,
    'use_base_offset': True,
    'use_residual_offset': True,
    'bound_residual': True,
    'warm_start_base': True,
}


def normalize_ablation(ablation: Optional[Dict] = None) -> Dict:
    cfg = DEFAULT_ABLATION.copy()
    if ablation:
        cfg.update(ablation)
    return cfg

class OGDConvBlock(nn.Module):
    """
    Deformable conv whose offsets are decomposed into:
      base   = structured rotation of the 3Ã—3 grid by Î¸  (parameter-free)
      delta  = small learned residual from Conv(cat[x, Î¸])

    Stability design:
      - offset_refine last layer is zero-initialized â†’ residual starts at 0
      - offsets soft-clamped via tanh Ã— max_delta (smooth, bounded gradients)
      - base_scale: learnable scalar, sigmoid-activated, initâ‰ˆ0 â†’ gradually
        activates geometric rotation as training progresses
    """

    MAX_DELTA = 3.0   # max pixel displacement for residual offsets

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ablation: Optional[Dict] = None,
    ):
        super().__init__()
        self.ablation = normalize_ablation(ablation)
        self.use_deformable = bool(self.ablation['use_deformable'])
        self.use_base_offset = bool(self.ablation['use_base_offset'])
        self.use_residual_offset = bool(self.ablation['use_residual_offset'])
        self.bound_residual = bool(self.ablation['bound_residual'])
        self.ks       = kernel_size
        self.n_points = kernel_size * kernel_size   # 9

        # Deformable conv weight/bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Residual offset network
        if self.use_deformable and self.use_residual_offset:
            self.offset_refine = nn.Sequential(
                nn.Conv2d(in_channels + 1, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2 * self.n_points, 1),
            )
        else:
            self.offset_refine = None
        # ZERO-INIT last layer â†’ residual offsets start at 0 at epoch 0
        if self.offset_refine is not None:
            nn.init.zeros_(self.offset_refine[-1].weight)
            nn.init.zeros_(self.offset_refine[-1].bias)

        # Learnable scale for base offsets.
        # sigmoid(-4) â‰ˆ 0.018 â†’ starts near 0 (identity), grows during training
        if self.use_deformable and self.use_base_offset:
            init_scale = -4.0 if self.ablation['warm_start_base'] else 8.0
            self.base_scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.base_scale = None

        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Canonical grid buffers
        mid = kernel_size // 2
        gy, gx = torch.meshgrid(
            torch.arange(-mid, mid + 1, dtype=torch.float32),
            torch.arange(-mid, mid + 1, dtype=torch.float32),
            indexing='ij',
        )
        self.register_buffer('grid_y', gy.reshape(1, self.n_points, 1, 1))
        self.register_buffer('grid_x', gx.reshape(1, self.n_points, 1, 1))

    def _base_offsets(self, theta: torch.Tensor) -> torch.Tensor:
        B, _, H, W = theta.shape
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        gy = self.grid_y
        gx = self.grid_x
        dy = gy * (cos_t - 1) - gx * sin_t   # [B, P, H, W]
        dx = gy * sin_t + gx * (cos_t - 1)
        offset = torch.stack([dy, dx], dim=2).view(B, self.n_points * 2, H, W)
        return offset

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if not self.use_deformable:
            out = F.conv2d(x, self.weight, self.bias, padding=self.ks // 2)
            return self.relu(self.bn(out))

        if theta.shape[-2:] != x.shape[-2:]:
            theta = F.interpolate(theta, size=x.shape[-2:], mode='bilinear', align_corners=False)

        B, _, H, W = x.shape
        if self.use_base_offset:
            scale = torch.sigmoid(self.base_scale)
            base = scale * self._base_offsets(theta)
        else:
            base = x.new_zeros(B, self.n_points * 2, H, W)

        if self.offset_refine is not None:
            delta = self.offset_refine(torch.cat([x, theta], dim=1))
            if self.bound_residual:
                delta = torch.tanh(delta) * self.MAX_DELTA
        else:
            delta = torch.zeros_like(base)
        offset = base + delta

        out = deform_conv2d(x, offset, self.weight, self.bias, padding=self.ks // 2)
        return self.relu(self.bn(out))


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class OGDDecoder(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 ablation: Optional[Dict] = None):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        fused_ch  = in_ch // 2 + skip_ch
        self.ogd  = OGDConvBlock(fused_ch, out_ch, ablation=ablation)
        self.norm = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                theta: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        if skip.shape[1] > 0:
            x = torch.cat([x, skip], dim=1)
        x = self.ogd(x, theta)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Full OGDNet  (flexible backbone)
# ---------------------------------------------------------------------------

class OGDNet(nn.Module):
    """
    Backbone options:
      'resnet18'      â†’ 5 levels, ch=[64, 64,128,256,512]
      'resnet34'      â†’ 5 levels, ch=[64, 64,128,256,512]
      'resnet50'      â†’ 5 levels, ch=[64,256,512,1024,2048]
      'resnet101'     â†’ 5 levels, ch=[64,256,512,1024,2048]
      'convnext_tiny' â†’ 4 levels, ch=[96,192,384,768]

    The decoder adapts automatically to backbone channel widths.
    """

    def __init__(
        self,
        backbone: str             = 'resnet34',
        pretrained: bool          = True,
        decoder_channels: List[int] = None,
        orient_mid: int           = 128,
        ablation: Optional[Dict]  = None,
    ):
        super().__init__()
        self.ablation = normalize_ablation(ablation)

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        # â”€â”€ Encoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        n_levels = 4 if backbone.startswith('convnext') else 5
        out_idx  = tuple(range(n_levels))

        self.encoder = timm.create_model(
            backbone, pretrained=pretrained,
            features_only=True, out_indices=out_idx,
        )
        enc_ch = list(self.encoder.feature_info.channels())

        # â”€â”€ Orientation head (on stride-16 level) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        orient_idx = 2 if backbone.startswith('convnext') else 3  # stride-16
        if self.ablation['use_orientation']:
            self.orient_head = OrientationHead(enc_ch[orient_idx], orient_mid)
        else:
            self.orient_head = None

        # â”€â”€ Decoder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        dc = decoder_channels

        if n_levels == 5:
            # ResNet-34/50: f0â€¦f4
            self.dec4 = OGDDecoder(enc_ch[4], enc_ch[3], dc[0], self.ablation)
            self.dec3 = OGDDecoder(dc[0],    enc_ch[2], dc[1], self.ablation)
            self.dec2 = OGDDecoder(dc[1],    enc_ch[1], dc[2], self.ablation)
            self.dec1 = OGDDecoder(dc[2],    enc_ch[0], dc[3], self.ablation)
            self.dec0 = OGDDecoder(dc[3],    0,         dc[4], self.ablation)
            self._n_stages = 5
        else:
            # ConvNeXt-Tiny: f0â€¦f3  (already at stride 4 for f0)
            self.dec3 = OGDDecoder(enc_ch[3], enc_ch[2], dc[0], self.ablation)
            self.dec2 = OGDDecoder(dc[0],    enc_ch[1], dc[1], self.ablation)
            self.dec1 = OGDDecoder(dc[1],    enc_ch[0], dc[2], self.ablation)
            self.dec0 = OGDDecoder(dc[2],    0,         dc[3], self.ablation)
            self._n_stages = 4

        out_ch = dc[3] if n_levels == 4 else dc[4]
        self.head = nn.Sequential(
            nn.Conv2d(out_ch, 1, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feats = self.encoder(x)

        # Convert any channels-last (NHWC) feature maps to NCHW.
        # Detect by comparing against the known channel count rather than by
        # H/W equality â€” the latter wrongly fires on non-square inputs
        # (e.g. 640Ã—360) where a valid NCHW map has unequal spatial dims.
        enc_ch = list(self.encoder.feature_info.channels())
        feats_nchw = []
        for f, c in zip(feats, enc_ch):
            if f.shape[1] != c and f.shape[-1] == c:
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        feats = feats_nchw

        if self._n_stages == 5:
            f0, f1, f2, f3, f4 = feats
            theta = self._make_theta(f3)
            d4 = self.dec4(f4, f3, theta)
            d3 = self.dec3(d4, f2, theta)
            d2 = self.dec2(d3, f1, theta)
            d1 = self.dec1(d2, f0, theta)
            dummy = torch.zeros(B, 0, d1.shape[2]*2, d1.shape[3]*2, device=x.device)
            d0 = self.dec0(d1, dummy, theta)
        else:
            f0, f1, f2, f3 = feats
            theta = self._make_theta(f2)
            d3 = self.dec3(f3, f2, theta)
            d2 = self.dec2(d3, f1, theta)
            d1 = self.dec1(d2, f0, theta)
            dummy = torch.zeros(B, 0, d1.shape[2]*2, d1.shape[3]*2, device=x.device)
            d0 = self.dec0(d1, dummy, theta)

        out = self.head(d0)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

    def _make_theta(self, feature: torch.Tensor) -> torch.Tensor:
        if self.orient_head is not None:
            return self.orient_head(feature)
        return feature.new_zeros(feature.shape[0], 1, feature.shape[2], feature.shape[3])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(config: Dict) -> OGDNet:
    cfg = config.get('model', config)
    return OGDNet(
        backbone         = cfg.get('backbone', 'resnet34'),
        pretrained       = cfg.get('pretrained', True),
        decoder_channels = cfg.get('decoder_channels', [256, 128, 64, 32, 16]),
        orient_mid       = cfg.get('orient_mid', 128),
        ablation         = cfg.get('ablation'),
    )


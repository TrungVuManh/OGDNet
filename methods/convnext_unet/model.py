"""
UNet with ConvNeXt pretrained encoder.
Liu et al. "A ConvNet for the 2020s." CVPR 2022.

Architecture:
  ConvNeXt encoder (timm: convnext_tiny) → 4 multi-scale features [strides 4, 8, 16, 32]
  UNet-style skip-connection decoder → binary mask
  Final bilinear upsample to input resolution (stride 4→1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Dict


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvNeXtUNet(nn.Module):
    """
    UNet with ConvNeXt encoder (4-level hierarchy at strides 4, 8, 16, 32).
    Output is upsampled to input resolution via bilinear interpolation.

    Args:
        encoder_name:     timm ConvNeXt model name
        pretrained:       load ImageNet weights
        out_channels:     number of output channels
        decoder_channels: per-stage output channels [d3, d2, d1, d0]
    """

    def __init__(
        self,
        encoder_name: str = 'convnext_tiny',
        pretrained: bool = True,
        out_channels: int = 1,
        decoder_channels: List[int] = [256, 128, 64, 32],
    ):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        enc_chs = self.encoder.feature_info.channels()
        # convnext_tiny: [96, 192, 384, 768] @ strides [4, 8, 16, 32]

        # dec3: up(f3) + f2  → d[0]   (16×16 → 32×32)
        # dec2: up(d0) + f1  → d[1]   (32×32 → 64×64)
        # dec1: up(d1) + f0  → d[2]   (64×64 → 128×128)
        # dec0: up(d2)       → d[3]   (128×128 → 256×256, no skip)
        # final bilinear to input_size (256×256 → 512×512)
        self.dec3 = DecoderBlock(enc_chs[3], enc_chs[2], decoder_channels[0])
        self.dec2 = DecoderBlock(decoder_channels[0], enc_chs[1], decoder_channels[1])
        self.dec1 = DecoderBlock(decoder_channels[1], enc_chs[0], decoder_channels[2])
        self.dec0 = DecoderBlock(decoder_channels[2], 0, decoder_channels[3])

        self.head = nn.Conv2d(decoder_channels[3], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        f0, f1, f2, f3 = self.encoder(x)

        d = self.dec3(f3, f2)
        d = self.dec2(d, f1)
        d = self.dec1(d, f0)
        d = self.dec0(d)

        out = self.head(d)
        return torch.sigmoid(F.interpolate(out, size=input_size, mode='bilinear', align_corners=True))


def build_model(config: Dict) -> ConvNeXtUNet:
    model_config = config.get('model', config)
    return ConvNeXtUNet(
        encoder_name=model_config.get('encoder_name', 'convnext_tiny'),
        pretrained=model_config.get('pretrained', True),
        out_channels=model_config.get('out_channels', 1),
        decoder_channels=model_config.get('decoder_channels', [256, 128, 64, 32]),
    )

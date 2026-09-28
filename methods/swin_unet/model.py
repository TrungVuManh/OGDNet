"""
Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
Cao et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List


class PatchExpand(nn.Module):
    """Patch expanding layer for upsampling."""

    def __init__(self, dim: int, dim_scale: int = 2):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = self.norm(x)
        return x


class SwinUnet(nn.Module):
    """Swin-Unet for medical image segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        embed_dim: int = 96,
        pretrained: bool = True,
        img_size: int = 640,
        dropout: float = 0.1
    ):
        super().__init__()

        # Swin Transformer encoder
        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            img_size=img_size,
            window_size=8
        )

        encoder_dims = self.encoder.feature_info.channels()

        # Decoder with skip connections
        self.up_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        decoder_dims = encoder_dims[::-1]  # Reverse order
        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i]
            out_dim = decoder_dims[i + 1]

            self.up_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.GELU(),
                nn.Dropout2d(dropout)  # Added dropout for regularization
            ))

            self.skip_convs.append(nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.GELU(),
                nn.Dropout2d(dropout)  # Added dropout for regularization
            ))

        # Final upsampling and head
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(encoder_dims[0], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(dropout / 2)  # Lower dropout before final head
        )

        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Ensure NCHW format (timm swin might return NHWC)
        features = [f.permute(0, 3, 1, 2).contiguous() if f.shape[-1] == self.encoder.feature_info.channels()[i] else f 
                   for i, f in enumerate(features)]

        # Decoder with skip connections
        x = features[-1]
        for i, (up_layer, skip_conv) in enumerate(zip(self.up_layers, self.skip_convs)):
            x = up_layer(x)
            skip = features[-(i + 2)]

            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([x, skip], dim=1)
            x = skip_conv(x)

        # Final upsampling
        x = self.final_up(x)

        # Head
        out = self.head(x)

        return torch.sigmoid(out)


def build_model(config: Dict) -> SwinUnet:
    model_config = config.get('model', config)
    return SwinUnet(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 1),
        embed_dim=model_config.get('embed_dim', 96),
        pretrained=model_config.get('pretrained', True),
        img_size=model_config.get('img_size', 640),
        dropout=model_config.get('dropout', 0.1)
    )

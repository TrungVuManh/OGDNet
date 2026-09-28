"""
CrackFormer: Transformer Network for Fine-Grained Crack Detection
Liu et al., ICCV 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict


class DirectionalConvModule(nn.Module):
    """Directional convolution module for crack-aware feature extraction."""

    def __init__(self, in_channels: int, kernel_size: int = 7):
        super().__init__()

        # Horizontal conv (1 x K)
        self.h_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=in_channels
        )

        # Vertical conv (K x 1)
        self.v_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=in_channels
        )

        # Diagonal convs
        self.d1_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.d2_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h = self.h_conv(x)
        v = self.v_conv(x)
        d1 = self.d1_conv(x)
        d2 = self.d2_conv(x)

        concat = torch.cat([h, v, d1, d2], dim=1)
        out = self.fusion(concat)
        return out + x  # Residual


class CrackAwareBlock(nn.Module):
    """Crack-aware transformer block with spatial pooling for large feature maps.

    For feature maps larger than max_spatial_size, pools down before
    self-attention and interpolates back to prevent OOM.
    """

    def __init__(self, dim: int, num_heads: int = 4, kernel_size: int = 7,
                 dropout: float = 0.1, max_spatial_size: int = 32):
        super().__init__()

        self.max_spatial_size = max_spatial_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.directional = DirectionalConvModule(dim, kernel_size)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.drop_path = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape

        # Pool down if spatial dims are too large for self-attention
        need_pool = H > self.max_spatial_size or W > self.max_spatial_size
        if need_pool:
            pool_h = min(H, self.max_spatial_size)
            pool_w = min(W, self.max_spatial_size)
            x_pooled = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
            _, _, pH, pW = x_pooled.shape
        else:
            x_pooled = x
            pH, pW = H, W

        # Attention on (possibly pooled) feature map
        x_flat = x_pooled.flatten(2).transpose(1, 2)  # (B, pH*pW, C)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + self.drop_path(attn_out)

        # Reshape back
        x_attn = x_flat.transpose(1, 2).reshape(B, C, pH, pW)

        # Interpolate back to original size if pooled
        if need_pool:
            x_attn = F.interpolate(x_attn, size=(H, W), mode='bilinear', align_corners=True)

        # Add residual from original input
        x = x + x_attn - (x_pooled if not need_pool else F.interpolate(x_pooled, size=(H, W), mode='bilinear', align_corners=True))

        # Directional convolution (at full resolution)
        x = self.directional(x)

        # MLP (at full resolution)
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.drop_path(self.mlp(self.norm3(x_flat)))
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)

        return x


class CrackFormer(nn.Module):
    """CrackFormer for crack detection."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        embed_dim: int = 64,
        depths: list = [2, 2, 6, 2],
        num_heads: list = [2, 4, 8, 16],
        directional_kernel: int = 7,
        pretrained: bool = True,
        img_size: int = 640,
        dropout: float = 0.1
    ):
        super().__init__()

        # Use Swin Transformer as backbone
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            img_size=img_size,
            window_size=8
        )

        encoder_dims = self.backbone.feature_info.channels()

        # Crack-aware enhancement at each scale
        # max_spatial_size=32 means attention is done on at most 32x32=1024 tokens
        self.crack_blocks = nn.ModuleList([
            CrackAwareBlock(dim, heads, directional_kernel, dropout=dropout, max_spatial_size=32)
            for dim, heads in zip(encoder_dims, num_heads)
        ])

        # Decoder with dropout
        self.decoder = nn.ModuleList()
        decoder_dims = encoder_dims[::-1]

        for i in range(len(decoder_dims) - 1):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(decoder_dims[i], decoder_dims[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ))

        # Final layers
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(encoder_dims[0], 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout / 2)
        )

        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        input_shape = x.shape[2:]

        # Backbone features
        features = self.backbone(x)

        # Ensure NCHW format (timm swin might return NHWC)
        encoder_channels = self.backbone.feature_info.channels()
        processed = []
        for i, f in enumerate(features):
            if f.dim() == 4 and f.shape[-1] == encoder_channels[i] and f.shape[1] != encoder_channels[i]:
                f = f.permute(0, 3, 1, 2).contiguous()
            processed.append(f)
        features = processed

        # Apply crack-aware blocks
        enhanced = []
        for feat, crack_block in zip(features, self.crack_blocks):
            enhanced.append(crack_block(feat))

        # Decoder
        x = enhanced[-1]
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x)
            skip = enhanced[-(i + 2)]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = x + skip

        # Final
        x = self.final_up(x)
        out = self.head(x)

        # Ensure output matches input spatial size
        if out.shape[2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)

        return torch.sigmoid(out)


def build_model(config: Dict) -> CrackFormer:
    model_config = config.get('model', config)
    return CrackFormer(
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 1),
        embed_dim=model_config.get('embed_dim', 64),
        depths=model_config.get('depths', [2, 2, 6, 2]),
        num_heads=model_config.get('num_heads', [2, 4, 8, 16]),
        directional_kernel=model_config.get('directional_kernel', 7),
        pretrained=model_config.get('pretrained', True),
        img_size=model_config.get('img_size', 640),
        dropout=model_config.get('dropout', 0.1)
    )

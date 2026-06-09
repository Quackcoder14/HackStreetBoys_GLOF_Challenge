"""
model_architecture.py – Custom U-Net Architecture
SNUC GLOFeagles 2026 Challenge

Defines the U-Net model with a custom convolutional encoder and decoder blocks.
Provides compliance with GLOFeagles submission guidelines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """Two ConvBnRelu blocks followed by MaxPool."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        out = self.pool(feat)
        return feat, out


class DecoderBlock(nn.Module):
    """Upsample 2× (bilinear) → concat skip → two ConvBnRelu blocks."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.drop  = nn.Dropout2d(p=dropout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:],
                                   mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class GlacialLakeUNet(nn.Module):
    """
    Custom U-Net segmentation model for glacial lake detection.
    """
    def __init__(self, pretrained: bool = False, num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(3, 32)      # s1: 32ch
        self.enc2 = EncoderBlock(32, 64)     # s2: 64ch
        self.enc3 = EncoderBlock(64, 128)    # s3: 128ch
        self.enc4 = EncoderBlock(128, 256)   # s4: 256ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnRelu(256, 512),
            ConvBnRelu(512, 512)
        )
        
        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256, dropout)
        self.dec3 = DecoderBlock(256, 128, 128, dropout)
        self.dec2 = DecoderBlock(128,  64,  64, dropout)
        self.dec1 = DecoderBlock( 64,  32,  32, dropout)
        
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        
        b = self.bottleneck(p4)
        
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        
        logits = self.head(d1)
        
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits

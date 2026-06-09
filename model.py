"""
model.py – Lightweight Custom U-Net
SNUC GLOFeagles 2026 Challenge

Uses a custom CNN encoder and decoder to avoid external dependencies like timm or torchvision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Custom U-Net
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss (operates on probabilities)."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = probs.view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        dice  = (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    0.6 × Dice + 0.4 × BCE  (with positive class up-weighting).
    Handles class imbalance: lakes are a minority of pixels.
    """
    def __init__(self, dice_weight: float = None, bce_weight: float = None,
                 pos_weight: float = 3.0):
        super().__init__()
        self.dice_w = dice_weight or config.DICE_WEIGHT
        self.bce_w  = bce_weight  or config.BCE_WEIGHT
        self.dice   = DiceLoss()
        self.bce    = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        loss_dice = self.dice(probs, targets)
        loss_bce  = self.bce(logits, targets)
        return self.dice_w * loss_dice + self.bce_w * loss_bce


# ─────────────────────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = False) -> GlacialLakeUNet:
    model = GlacialLakeUNet(
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
    )
    return model.to(config.DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    model = build_model(pretrained=False)
    x     = torch.randn(2, 3, 512, 512).to(config.DEVICE)
    with torch.no_grad():
        logits = model(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 1, 512, 512), f"Shape mismatch: {logits.shape}"

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
    print("Model sanity check PASSED [OK]")


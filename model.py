"""
model.py
========
DoseNet3D: an asymmetric U-Net for 3D dose prediction.

    Encoder: 3D convolutions (true 3D context) with residual
             squeeze-and-excitation blocks (RSEM), downsampling on all
             three axes.
    Decoder: 2D deformable convolutions applied per-axial-slice ("2.5D" --
             depth folded into the batch dimension), wrapped in the same
             residual/SE structure (RDSEM), meant to preserve sharp
             in-plane PTV/OAR dose-falloff edges that a true 3D conv would
             tend to blur across slices. Skip connections concatenate
             matching-resolution encoder features (U-Net style).

Shapes (in_channels=3 default): [B,3,256,256,80] -> [B,1,256,256,80]
    stem/enc1: 256^2x80 -> 128^2x40, channels 24
    enc2:      128^2x40 -> 64^2x20,  channels 48
    enc3:      64^2x20  -> 32^2x10,  channels 96 (bottleneck)
    dec1:      32^2x10  -> 64^2x20,  channels 48 (+ enc2 skip)
    dec2:      64^2x20  -> 128^2x40, channels 24 (+ enc1 skip)
    dec3:      128^2x40 -> 256^2x80, channels 24 (+ raw-input skip)
    head:      1x1x1 conv -> 1 channel, ReLU (dose >= 0)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


# --------------------------------------------------------------------------- #
# Squeeze-and-Excitation
# --------------------------------------------------------------------------- #

class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        s = self.pool(x).view(b, c)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s)).view(b, c, 1, 1, 1)
        return x * s


class SEBlock2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        s = self.pool(x).view(b, c)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s)).view(b, c, 1, 1)
        return x * s


# --------------------------------------------------------------------------- #
# RSEM: Residual Squeeze-and-Excitation Module (3D, encoder)
# --------------------------------------------------------------------------- #

class RSEM(nn.Module):
    """x_out = ReLU(x + SE(Conv-IN-ReLU-Conv-IN(x))), channels unchanged."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock3D(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        return self.act(x + out)


# --------------------------------------------------------------------------- #
# RDSEM: Residual Deformable Squeeze-and-Excitation Module (2D, decoder)
# --------------------------------------------------------------------------- #

class DeformConvBlock2D(nn.Module):
    """A single modulated (DCNv2-style) 2D deformable conv: a small offset
    sub-network predicts per-location (dx, dy) sampling offsets, and a
    parallel sigmoid mask sub-network predicts a per-location modulation
    weight, both consumed by torchvision.ops.DeformConv2d."""

    def __init__(self, channels: int, kernel_size: int = 3, deform_groups: int = 1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size * deform_groups
        mask_channels = kernel_size * kernel_size * deform_groups
        self.offset_conv = nn.Conv2d(channels, offset_channels, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(channels, mask_channels, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(channels, channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2)
        # Zero-init the offset/mask heads so this block starts out equivalent
        # to a regular conv (offsets=0, mask=sigmoid(0)=0.5 constant) --
        # standard trick for stabilizing deformable-conv training.
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.deform_conv(x, offset, mask)


class RDSEM(nn.Module):
    """Same residual + SE structure as RSEM, but both internal convs are 2D
    deformable convs. Expects a [N, C, H, W] tensor where N = B*D -- i.e.
    the caller has folded the depth axis into the batch dimension so each
    axial slice is deformed independently ("2.5D")."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.dconv1 = DeformConvBlock2D(channels)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.dconv2 = DeformConvBlock2D(channels)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock2D(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.dconv1(x)))
        out = self.norm2(self.dconv2(out))
        out = self.se(out)
        return self.act(x + out)


def fold_depth_into_batch(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """[B, C, H, W, D] -> [B*D, C, H, W]"""
    b, c, h, w, d = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w), b, d


def unfold_batch_to_depth(x: torch.Tensor, b: int, d: int) -> torch.Tensor:
    """[B*D, C, H, W] -> [B, C, H, W, D]"""
    n, c, h, w = x.shape
    return x.reshape(b, d, c, h, w).permute(0, 2, 3, 4, 1)


# --------------------------------------------------------------------------- #
# Encoder / decoder stages
# --------------------------------------------------------------------------- #

class EncoderStage(nn.Module):
    """Strided 3D conv (does channel projection + downsampling in one op) ->
    InstanceNorm -> ReLU -> RSEM refinement at the new resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.rsem = RSEM(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.down(x)))
        return self.rsem(x)


class DecoderStage(nn.Module):
    """Transposed 3D conv upsample (channel projection + upsampling in one
    op) -> concat skip -> 1x1x1 channel reduction -> fold depth into batch
    -> RDSEM (2D deformable) -> unfold back to 3D."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        concat_ch = out_ch + skip_ch
        self.reduce = nn.Conv3d(concat_ch, out_ch, kernel_size=1)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.ReLU(inplace=True)
        self.rdsem = RDSEM(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.reduce(x)))
        x2d, b, d = fold_depth_into_batch(x)
        x2d = self.rdsem(x2d)
        return unfold_batch_to_depth(x2d, b, d)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class DoseNet3D(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: Tuple[int, int, int] = (24, 48, 96)):
        super().__init__()
        self.in_channels = in_channels
        c1, c2, c3 = base_channels

        self.enc1 = EncoderStage(in_channels, c1)   # 256^2x80 -> 128^2x40
        self.enc2 = EncoderStage(c1, c2)              # 128^2x40 -> 64^2x20
        self.enc3 = EncoderStage(c2, c3)              # 64^2x20  -> 32^2x10 (bottleneck)

        self.dec1 = DecoderStage(c3, skip_ch=c2, out_ch=c2)             # -> 64^2x20
        self.dec2 = DecoderStage(c2, skip_ch=c1, out_ch=c1)             # -> 128^2x40
        self.dec3 = DecoderStage(c1, skip_ch=in_channels, out_ch=c1)    # -> 256^2x80 (skip = raw input)

        self.out_conv = nn.Conv3d(c1, 1, kernel_size=1)
        self.out_act = nn.ReLU(inplace=True)   # dose can't be negative

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        bottleneck = self.enc3(s2)

        d1 = self.dec1(bottleneck, skip=s2)
        d2 = self.dec2(d1, skip=s1)
        d3 = self.dec3(d2, skip=x)

        return self.out_act(self.out_conv(d3))


def print_model_summary(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DoseNet3D parameter count: {total:,} total, {trainable:,} trainable")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:10s} {n:>12,} params")


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_cohort
    from data_pipeline import PreprocessConfig

    cfg = PreprocessConfig()
    model = DoseNet3D(in_channels=3)
    print_model_summary(model)

    cohort = generate_synthetic_cohort(n_patients=1, cfg=cfg)
    batch = torch.from_numpy(
        __import__("numpy").stack([ex.input for ex in cohort], axis=0)
    )
    print("\ninput batch shape:", tuple(batch.shape))

    model.eval()
    with torch.no_grad():
        out = model(batch)
    print("output batch shape:", tuple(out.shape))
    print("output min/max:", out.min().item(), out.max().item())
    assert out.shape == (batch.shape[0], 1, 256, 256, 80)
    assert (out >= 0).all()
    print("FORWARD PASS OK")

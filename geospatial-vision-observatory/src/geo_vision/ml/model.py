from __future__ import annotations

from typing import Any, cast


def _groups(channels: int) -> int:
    for value in (8, 4, 2):
        if channels % value == 0:
            return value
    return 1


def build_model(in_channels: int = 6, classes: int = 11, base_channels: int = 20) -> Any:
    """Build the flagship compact U-Net lazily so base installs do not require torch."""

    import torch
    from torch import nn

    class ConvNormAct(nn.Module):
        def __init__(self, source: int, target: int, *, stride: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(source, target, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(_groups(target), target),
                nn.SiLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.block(x))

    class ResidualSeparable(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
            )
            self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.norm = nn.GroupNorm(_groups(channels), channels)
            self.activation = nn.SiLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.norm(x)
            return cast(torch.Tensor, self.activation(x + residual))

    class EncoderStage(nn.Module):
        def __init__(self, source: int, target: int, *, downsample: bool) -> None:
            super().__init__()
            self.projection = ConvNormAct(source, target, stride=2 if downsample else 1)
            self.refine = ResidualSeparable(target)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.refine(self.projection(x)))

    class DecoderStage(nn.Module):
        def __init__(self, source: int, skip: int, target: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(source, target, kernel_size=2, stride=2)
            self.fuse = nn.Sequential(
                ConvNormAct(target + skip, target),
                ResidualSeparable(target),
            )

        def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
            x = self.up(x)
            if x.shape[-2:] != skip_x.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip_x.shape[-2:], mode="bilinear", align_corners=False
                )
            return cast(torch.Tensor, self.fuse(torch.cat((x, skip_x), dim=1)))

    class EfficientMultispectralUNetImpl(nn.Module):
        """Small six-band segmentation network designed for reproducible local training."""

        def __init__(self) -> None:
            super().__init__()
            b = base_channels
            self.enc0 = EncoderStage(in_channels, b, downsample=False)
            self.enc1 = EncoderStage(b, b * 2, downsample=True)
            self.enc2 = EncoderStage(b * 2, b * 4, downsample=True)
            self.enc3 = EncoderStage(b * 4, b * 8, downsample=True)
            self.bottleneck = nn.Sequential(
                ConvNormAct(b * 8, b * 12, stride=2),
                ResidualSeparable(b * 12),
                nn.Dropout2d(0.10),
            )
            self.dec3 = DecoderStage(b * 12, b * 8, b * 8)
            self.dec2 = DecoderStage(b * 8, b * 4, b * 4)
            self.dec1 = DecoderStage(b * 4, b * 2, b * 2)
            self.dec0 = DecoderStage(b * 2, b, b)
            self.head = nn.Conv2d(b, classes, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e0 = self.enc0(x)
            e1 = self.enc1(e0)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            x = self.bottleneck(e3)
            x = self.dec3(x, e3)
            x = self.dec2(x, e2)
            x = self.dec1(x, e1)
            x = self.dec0(x, e0)
            return cast(torch.Tensor, self.head(x))

    return EfficientMultispectralUNetImpl()


# Friendly public factory alias used by documentation and importers.
EfficientMultispectralUNet = build_model

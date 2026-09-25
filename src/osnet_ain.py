"""OSNet-AIN-x1.0 compatible with official deep-person-reid checkpoints.

This module implements the architecture only. It never downloads weights; the
checkpoint is supplied through ``configs/config.yaml``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .osnet import ChannelGate, Conv1x1, LightConv3x3


class ConvLayerIN(nn.Module):
    """Convolution followed by InstanceNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=False
        )
        # The attribute is intentionally named ``bn`` to match official keys.
        self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1LinearAIN(nn.Module):
    """Linear 1x1 convolution with optional BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int, bn: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConvStream(nn.Module):
    """A sequence of lightweight convolutions at one receptive-field scale."""

    def __init__(self, in_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("LightConvStream depth must be at least one.")
        layers: list[nn.Module] = [LightConv3x3(in_channels, out_channels)]
        layers.extend(LightConv3x3(out_channels, out_channels) for _ in range(depth - 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OSBlock(nn.Module):
    """Standard omni-scale residual block used inside OSNet-AIN."""

    def __init__(self, in_channels: int, out_channels: int,
                 reduction: int = 4, streams: int = 4) -> None:
        super().__init__()
        if streams < 1 or out_channels % reduction != 0:
            raise ValueError("Invalid OSBlock dimensions.")
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList([
            LightConvStream(mid_channels, mid_channels, depth)
            for depth in range(1, streams + 1)
        ])
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1LinearAIN(mid_channels, out_channels)
        self.downsample = (
            Conv1x1LinearAIN(in_channels, out_channels)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = sum((self.gate(stream(x1)) for stream in self.conv2), start=torch.zeros_like(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity, inplace=True)


class OSBlockINin(nn.Module):
    """Omni-scale block with InstanceNorm inside its residual branch."""

    def __init__(self, in_channels: int, out_channels: int,
                 reduction: int = 4, streams: int = 4) -> None:
        super().__init__()
        if streams < 1 or out_channels % reduction != 0:
            raise ValueError("Invalid OSBlockINin dimensions.")
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList([
            LightConvStream(mid_channels, mid_channels, depth)
            for depth in range(1, streams + 1)
        ])
        self.gate = ChannelGate(mid_channels)
        # Official AIN uses no BatchNorm here, then applies InstanceNorm.
        self.conv3 = Conv1x1LinearAIN(mid_channels, out_channels, bn=False)
        self.downsample = (
            Conv1x1LinearAIN(in_channels, out_channels)
            if in_channels != out_channels else None
        )
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = sum((self.gate(stream(x1)) for stream in self.conv2), start=torch.zeros_like(x1))
        x3 = self.IN(self.conv3(x2))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity, inplace=True)


class OSNetAIN(nn.Module):
    """OSNet-AIN-x1.0 feature extractor."""

    def __init__(self, num_classes: int = 1, feature_dim: int = 512) -> None:
        super().__init__()
        channels = [64, 256, 384, 512]
        self.conv1 = ConvLayerIN(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = self._make_layer(
            [OSBlockINin, OSBlockINin], channels[0], channels[1]
        )
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = self._make_layer(
            [OSBlock, OSBlockINin], channels[1], channels[2]
        )
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = self._make_layer(
            [OSBlockINin, OSBlock], channels[2], channels[3]
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self._init_params()

    @staticmethod
    def _make_layer(blocks: list[type[nn.Module]], in_channels: int,
                    out_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = [blocks[0](in_channels, out_channels)]
        layers.extend(block(out_channels, out_channels) for block in blocks[1:])
        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        return self.conv5(self.conv4(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.global_avgpool(self.featuremaps(x)).view(x.size(0), -1)
        features = self.fc(features)
        if not self.training:
            return features
        return self.classifier(features)


def osnet_ain_x1_0(num_classes: int = 1) -> OSNetAIN:
    """Construct OSNet-AIN-x1.0 without loading or downloading weights."""
    return OSNetAIN(num_classes=num_classes, feature_dim=512)

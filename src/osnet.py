"""OSNet-x1.0 architecture compatible with deep-person-reid checkpoints.

The implementation is local so constructing the network never contacts a model
hub. Only the weight path supplied in config.yaml is loaded. Architecture adapted
from the MIT-licensed KaiyangZhou/deep-person-reid OSNet implementation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(ConvLayer):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels, 1)


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16,
                 layer_norm: bool = False) -> None:
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced, 1)
        self.norm = nn.LayerNorm((reduced, 1, 1)) if layer_norm else None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc1(self.global_avgpool(x))
        if self.norm is not None:
            gate = self.norm(gate)
        gate = self.fc2(self.relu(gate))
        return x * torch.sigmoid(gate)


class OSBlock(nn.Module):
    """Omni-scale residual block used by OSNet."""

    def __init__(self, in_channels: int, out_channels: int, IN: bool = False) -> None:
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels)
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels)
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels), LightConv3x3(mid_channels, mid_channels)
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if IN else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1))
        x2 = x2 + self.gate(self.conv2b(x1))
        x2 = x2 + self.gate(self.conv2c(x1))
        x2 = x2 + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out, inplace=True)


class OSNet(nn.Module):
    def __init__(self, num_classes: int = 1, feature_dim: int = 512) -> None:
        super().__init__()
        channels = [64, 256, 384, 512]
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(2, channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(2, channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(2, channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self._init_params()

    @staticmethod
    def _make_layer(blocks: int, in_channels: int, out_channels: int,
                    reduce_spatial_size: bool) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(in_channels, out_channels)]
        layers.extend(OSBlock(out_channels, out_channels) for _ in range(1, blocks))
        if reduce_spatial_size:
            # Preserve torchreid's nested Sequential and therefore its checkpoint keys
            # (for example, conv2.2.0.conv.weight).
            layers.append(nn.Sequential(
                Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)
            ))
        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        features = self.global_avgpool(x).view(x.size(0), -1)
        features = self.fc(features)
        if not self.training:
            return features
        return self.classifier(features)


def osnet_x1_0(num_classes: int = 1) -> OSNet:
    """Construct OSNet-x1.0 without loading or downloading weights."""
    return OSNet(num_classes=num_classes, feature_dim=512)

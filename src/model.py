"""MedicalNet-style ResNet-18 backbone with a new four-class classification head.

Backbone adapted from Tencent/MedicalNet (MIT; see THIRD_PARTY_NOTICES.md).
The later stages retain MedicalNet's dilation, not a torchvision video architecture.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ShortcutA(nn.Module):
    """Parameter-free subsampling and zero-padding; preserve the autograd graph."""

    def __init__(self, channels, stride):
        super().__init__()
        self.channels, self.stride = channels, stride

    def forward(self, x):
        x = F.avg_pool3d(x, kernel_size=1, stride=self.stride)
        padding = x.new_zeros(x.shape[0], self.channels - x.shape[1], *x.shape[2:])
        return torch.cat((x, padding), dim=1)


class BasicBlock(nn.Module):
    """Two 3D convolutions plus an identity or channel-matching residual path."""

    def __init__(self, in_channels, channels, stride=1, dilation=1, shortcut="A"):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(
            channels, channels, 3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = (
                ShortcutA(channels, stride)
                if shortcut == "A"
                else nn.Sequential(
                    nn.Conv3d(in_channels, channels, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(channels),
                )
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + residual)


class ResNet18(nn.Module):
    """Four-class MedicalNet backbone. Reduced width is for synthetic smoke tests only."""

    def __init__(self, num_classes=4, base_channels=64, shortcut="A"):
        super().__init__()
        if shortcut not in {"A", "B"} or base_channels < 1 or num_classes != 4:
            raise ValueError("Invalid architecture settings")
        self.base_channels, self.shortcut = base_channels, shortcut
        c = base_channels
        self.conv1 = nn.Conv3d(1, c, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._stage(c, c, 1, 1)
        self.layer2 = self._stage(c, c * 2, 2, 1)
        self.layer3 = self._stage(c * 2, c * 4, 1, 2)
        self.layer4 = self._stage(c * 4, c * 8, 1, 4)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(c * 8, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _stage(self, in_channels, channels, stride, dilation):
        return nn.Sequential(
            BasicBlock(in_channels, channels, stride, dilation, self.shortcut),
            BasicBlock(channels, channels, 1, dilation, self.shortcut),
        )

    def get_layer4_activations(self, x):
        """Return spatial features with gradients attached for downstream explanations."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def forward_features(self, x):
        """Return [batch, 512] pooled features for the full-width model."""
        return self.avgpool(self.get_layer4_activations(x)).flatten(1)

    def forward_head(self, features):
        return self.fc(features)

    def forward_from_activations(self, activations):
        """Continue the classifier from spatial layer-4 activations without detaching."""
        return self.fc(self.avgpool(activations).flatten(1))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def build_model(config):
    return ResNet18(config["num_classes"], config["base_channels"], config["shortcut"])

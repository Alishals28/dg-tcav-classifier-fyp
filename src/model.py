"""MedicalNet-style ResNet-18 backbone with a new four-class classification head.

Backbone adapted from Tencent/MedicalNet (MIT; see THIRD_PARTY_NOTICES.md).
The later stages retain MedicalNet's dilation, not a torchvision video architecture.
"""

import torch
from torch import nn
from torch.nn import functional as F


def make_norm(norm: str, channels: int) -> nn.Module:
    """BatchNorm by default; GroupNorm/InstanceNorm do not depend on the batch size."""
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "group":
        return nn.GroupNorm(min(32, channels // 4), channels)
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"Unknown norm: {norm}")


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

    def __init__(self, in_channels, channels, stride=1, dilation=1, shortcut="A", norm="batch"):
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
        self.bn1 = make_norm(norm, channels)
        self.conv2 = nn.Conv3d(
            channels, channels, 3, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = make_norm(norm, channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = (
                ShortcutA(channels, stride)
                if shortcut == "A"
                else nn.Sequential(
                    nn.Conv3d(in_channels, channels, 1, stride=stride, bias=False),
                    make_norm(norm, channels),
                )
            )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + residual)


class ResNet18(nn.Module):
    """Four-class MedicalNet backbone. Reduced width is for synthetic smoke tests only."""

    def __init__(self, num_classes=4, base_channels=64, shortcut="A", norm="batch", dropout=0.0):
        super().__init__()
        if shortcut not in {"A", "B"} or base_channels < 1 or num_classes != 4:
            raise ValueError("Invalid architecture settings")
        self.base_channels, self.shortcut, self.norm = base_channels, shortcut, norm
        c = base_channels
        self.feature_dim = c * 8
        self.conv1 = nn.Conv3d(1, c, 7, stride=2, padding=3, bias=False)
        self.bn1 = make_norm(norm, c)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._stage(c, c, 1, 1)
        self.layer2 = self._stage(c, c * 2, 2, 1)
        self.layer3 = self._stage(c * 2, c * 4, 1, 2)
        self.layer4 = self._stage(c * 4, c * 8, 1, 4)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c * 8, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)) and module.weight is not None:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _stage(self, in_channels, channels, stride, dilation):
        return nn.Sequential(
            BasicBlock(in_channels, channels, stride, dilation, self.shortcut, self.norm),
            BasicBlock(channels, channels, 1, dilation, self.shortcut, self.norm),
        )

    def get_layer4_activations(self, x):
        """Return spatial features with gradients attached for downstream explanations."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def forward_features(self, x):
        """Return [batch, 512] pooled features for the full-width model."""
        return self.features_from_activations(self.get_layer4_activations(x))

    def features_from_activations(self, activations):
        return self.avgpool(activations).flatten(1)

    def forward_head(self, features):
        return self.fc(self.dropout(features))

    def forward_from_activations(self, activations):
        """Continue the classifier from spatial layer-4 activations without detaching."""
        return self.forward_head(self.features_from_activations(activations))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class ClassifierCNN(nn.Module):
    """Four-block dilated 3D CNN with a spatially flattened feature bottleneck."""

    def __init__(
        self,
        in_channels=1,
        num_classes=4,
        expansion=8,
        feature_dim=1024,
        nhid=512,
        norm_type="instance",
        input_shape=(128, 128, 128),
    ):
        super().__init__()
        if num_classes != 4 or norm_type not in {"instance", "batch"}:
            raise ValueError("ClassifierCNN requires four classes and a supported norm")
        if in_channels < 1 or expansion < 1 or feature_dim < 1 or nhid < 1:
            raise ValueError("ClassifierCNN dimensions must be positive")
        if len(input_shape) != 3 or any(int(size) < 1 for size in input_shape):
            raise ValueError("input_shape must contain three positive dimensions")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.norm_type = norm_type
        self.input_shape = tuple(int(size) for size in input_shape)
        norm = nn.InstanceNorm3d if norm_type == "instance" else nn.BatchNorm3d

        def make_backbone():
            return nn.Sequential(
                nn.Conv3d(in_channels, 4 * expansion, kernel_size=1),
                norm(4 * expansion, affine=True),
                nn.ReLU(inplace=False),
                nn.MaxPool3d(3, stride=2),
                nn.Conv3d(4 * expansion, 32 * expansion, kernel_size=3, dilation=2),
                norm(32 * expansion, affine=True),
                nn.ReLU(inplace=False),
                nn.MaxPool3d(3, stride=2),
                nn.Conv3d(
                    32 * expansion,
                    64 * expansion,
                    kernel_size=5,
                    padding=2,
                    dilation=2,
                ),
                norm(64 * expansion, affine=True),
                nn.ReLU(inplace=False),
                nn.MaxPool3d(3, stride=2),
                nn.Conv3d(
                    64 * expansion,
                    64 * expansion,
                    kernel_size=3,
                    padding=1,
                    dilation=2,
                ),
                norm(64 * expansion, affine=True),
                nn.ReLU(inplace=False),
                nn.MaxPool3d(5, stride=2),
            )

        self.conv = make_backbone()
        # Infer the flatten dimension without executing the very expensive 3D convolutions.
        with torch.random.fork_rng(devices=[]):
            with torch.device("meta"):
                shape_backbone = make_backbone()
                dummy = torch.zeros(1, in_channels, *self.input_shape, device="meta")
                flat_dim = shape_backbone(dummy).flatten(1).shape[1]
        if flat_dim < 1:
            raise ValueError("input_shape is too small for ClassifierCNN")

        self.fc6 = nn.Sequential(
            nn.Linear(flat_dim, feature_dim),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, nhid),
            nn.ReLU(inplace=False),
            nn.Linear(nhid, num_classes),
        )
        self.flat_dim = flat_dim
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def get_layer4_activations(self, x):
        return self.conv(x)

    def features_from_activations(self, activations):
        return self.fc6(activations.flatten(1))

    def forward_features(self, x):
        return self.features_from_activations(self.get_layer4_activations(x))

    def forward_head(self, features):
        return self.classifier(features)

    def forward_from_activations(self, activations):
        return self.forward_head(self.features_from_activations(activations))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def build_model(config, input_shape=None):
    architecture = config.get("architecture", "ResNet18")
    if architecture == "ResNet18":
        return ResNet18(
            config["num_classes"],
            config["base_channels"],
            config["shortcut"],
            config.get("norm", "batch"),
            config.get("dropout", 0.0),
        )
    if architecture == "ClassifierCNN":
        return ClassifierCNN(
            in_channels=config.get("in_channels", 1),
            num_classes=config["num_classes"],
            expansion=config.get("expansion", 8),
            feature_dim=config.get("feature_dim", 1024),
            nhid=config.get("nhid", 512),
            norm_type=config.get("norm_type", "instance"),
            input_shape=input_shape or config.get("input_shape", (128, 128, 128)),
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")

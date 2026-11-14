"""3D MRI Classifier architectures: MedicalNet ResNet-18, Hybrid MedicalNet Inception-ResNet-3D, and 3D Inception.

Includes:
- Configurable Dropout (e.g. 0.5) before the classification head
- Configurable 3D Pooling (`gap`, `attention`, `roi_attention`) to prevent 16^3 Global Average Pooling
  from washing out localized medial temporal / hippocampal atrophy
- Full MedicalNet pretrained weight compatibility for both `ResNet18` (100% backbone) and
  `InceptionResNet3D` / `HybridInceptionResNet3D` (pretrained MedicalNet stem `conv1`..`layer3` +
  multi-scale 3D Inception-ResNet `layer4`)
- Unified 512-dimensional bottleneck interface (`get_layer4_activations`, `forward_features`,
  `features_from_activations`, `forward_head`, `forward_from_activations`) for live-gradient TCAV.
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


class InceptionBlock3D(nn.Module):
    """Multi-scale 3D Inception module with optional residual skip connection (Inception-ResNet).

    Extracts multi-scale anatomical biomarkers in parallel:
    - Branch 1: 1x1x1 local voxel & cortical ribbon thickness features
    - Branch 2: 1x1x1 -> 3x3x3 medium subcortical features (hippocampus, amygdala)
    - Branch 3: 1x1x1 -> dilated 3x3x3 (5x5x5 receptive field for ventricles & temporal horns)
    - Branch 4: 3x3x3 MaxPool -> 1x1x1 high-contrast boundary features
    """

    def __init__(self, in_channels, out_channels, stride=1, use_residual=True):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("InceptionBlock3D out_channels must be divisible by 4")
        branch_ch = out_channels // 4
        self.use_residual = use_residual

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=False),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                branch_ch, branch_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=False),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=False),
            nn.Conv3d(
                branch_ch,
                branch_ch,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=False),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=stride, padding=1),
            nn.Conv3d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=False),
        )

        self.downsample = None
        if use_residual and (stride != 1 or in_channels != out_channels):
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1
        )
        if self.use_residual:
            residual = x if self.downsample is None else self.downsample(x)
            out = self.relu(out + residual)
        return out


class SpatialAttentionPool3D(nn.Module):
    """Replaces flat 16^3 Global Average Pooling with learned 3D spatial attention
    and an optional MNI152 medial-temporal/subcortical Gaussian anatomical prior."""

    def __init__(self, channels, use_roi_prior=False):
        super().__init__()
        self.use_roi_prior = use_roi_prior
        hidden = max(channels // 8, 4)
        self.attn_conv = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=False),
            nn.Conv3d(hidden, 1, kernel_size=1, bias=True),
        )

    @staticmethod
    def _medial_temporal_prior(shape, device, dtype):
        """Construct a smooth 3D anatomical prior emphasizing the central subcortical /
        medial temporal region (hippocampus, entorhinal cortex, ventricles) in MNI space."""
        _, _, d, h, w = shape
        z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype).view(1, 1, d, 1, 1)
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, 1, h, 1)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, 1, w)
        # Emphasize central/inferior-medial regions while keeping non-zero weight everywhere
        prior = 0.5 + 0.5 * torch.exp(-((z / 0.65) ** 2 + (y / 0.65) ** 2 + (x / 0.70) ** 2))
        return prior

    def forward(self, activations):
        # activations: [B, C, D, H, W]
        b, c, d, h, w = activations.shape
        logits = self.attn_conv(activations)  # [B, 1, D, H, W]
        if self.use_roi_prior:
            prior = self._medial_temporal_prior(activations.shape, activations.device, activations.dtype)
            logits = logits + torch.log(prior)
        weights = torch.softmax(logits.flatten(2), dim=-1).view(b, 1, d, h, w)
        attn_pooled = (activations * weights).sum(dim=(2, 3, 4))
        gap_pooled = activations.mean(dim=(2, 3, 4))
        # Residual blend of 50% flat GAP + 50% spatial attention preserves gradient stability
        return 0.5 * gap_pooled + 0.5 * attn_pooled


def _build_pooler(pooling, feature_dim):
    if pooling == "gap":
        return None
    if pooling == "attention":
        return SpatialAttentionPool3D(feature_dim, use_roi_prior=False)
    if pooling == "roi_attention":
        return SpatialAttentionPool3D(feature_dim, use_roi_prior=True)
    raise ValueError(f"Unsupported pooling mode: {pooling}. Choose gap, attention, or roi_attention")


class ResNet18(nn.Module):
    """Four-class MedicalNet 3D ResNet-18 with configurable Dropout and Spatial Attention Pooling."""

    def __init__(
        self,
        num_classes=4,
        base_channels=64,
        shortcut="A",
        dropout=0.0,
        pooling="gap",
    ):
        super().__init__()
        if (
            shortcut not in {"A", "B"}
            or base_channels < 1
            or num_classes != 4
            or not (0.0 <= float(dropout) < 1.0)
        ):
            raise ValueError("Invalid architecture settings")
        self.base_channels, self.shortcut = base_channels, shortcut
        self.dropout_rate = float(dropout)
        self.pooling_mode = pooling
        c = base_channels
        self.feature_dim = c * 8
        self.conv1 = nn.Conv3d(1, c, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._stage(c, c, 1, 1)
        self.layer2 = self._stage(c, c * 2, 2, 1)
        self.layer3 = self._stage(c * 2, c * 4, 1, 2)
        self.layer4 = self._stage(c * 4, c * 8, 1, 4)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.spatial_pool = _build_pooler(pooling, self.feature_dim)
        self.dropout = (
            nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        )
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
        return self.features_from_activations(self.get_layer4_activations(x))

    def features_from_activations(self, activations):
        if self.spatial_pool is not None:
            return self.spatial_pool(activations)
        return self.avgpool(activations).flatten(1)

    def forward_head(self, features):
        return self.fc(self.dropout(features))

    def forward_from_activations(self, activations):
        """Continue the classifier from spatial layer-4 activations without detaching."""
        return self.forward_head(self.features_from_activations(activations))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class HybridInceptionResNet3D(nn.Module):
    """Supervisor Hybrid Architecture: MedicalNet-compatible 3D ResNet stem (`conv1`..`layer3`)
    coupled with Multi-Scale 3D Inception-ResNet blocks in `layer4`.

    When `pretrained: true`, `load_medicalnet` loads 100% of `conv1`, `bn1`, `layer1`, `layer2`,
    and `layer3` directly from `resnet_18_23dataset.pth` while `layer4` extracts multi-scale
    ($1^3$, $3^3$, dilated $5^3$ RF, and MaxPool) features with residual skip connections.
    """

    def __init__(
        self,
        num_classes=4,
        base_channels=64,
        shortcut="A",
        dropout=0.5,
        pooling="roi_attention",
    ):
        super().__init__()
        if (
            shortcut not in {"A", "B"}
            or base_channels < 4
            or base_channels % 4 != 0
            or num_classes != 4
            or not (0.0 <= float(dropout) < 1.0)
        ):
            raise ValueError("Invalid HybridInceptionResNet3D settings")
        self.base_channels, self.shortcut = base_channels, shortcut
        self.dropout_rate = float(dropout)
        self.pooling_mode = pooling
        c = base_channels
        self.feature_dim = c * 8

        # MedicalNet-compatible ResNet stem and stages 1-3 (84/108 backbone tensors)
        self.conv1 = nn.Conv3d(1, c, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._resnet_stage(c, c, 1, 1)
        self.layer2 = self._resnet_stage(c, c * 2, 2, 1)
        self.layer3 = self._resnet_stage(c * 2, c * 4, 1, 2)

        # Stage 4: Multi-scale 3D Inception-ResNet blocks (stride 2 -> 8^3 grid, 512 channels)
        self.layer4 = nn.Sequential(
            InceptionBlock3D(c * 4, c * 8, stride=2, use_residual=True),
            InceptionBlock3D(c * 8, c * 8, stride=1, use_residual=True),
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.spatial_pool = _build_pooler(pooling, self.feature_dim)
        self.dropout = (
            nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        )
        self.fc = nn.Linear(self.feature_dim, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _resnet_stage(self, in_channels, channels, stride, dilation):
        return nn.Sequential(
            BasicBlock(in_channels, channels, stride, dilation, self.shortcut),
            BasicBlock(channels, channels, 1, dilation, self.shortcut),
        )

    def get_layer4_activations(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def forward_features(self, x):
        return self.features_from_activations(self.get_layer4_activations(x))

    def features_from_activations(self, activations):
        if self.spatial_pool is not None:
            return self.spatial_pool(activations)
        return self.avgpool(activations).flatten(1)

    def forward_head(self, features):
        return self.fc(self.dropout(features))

    def forward_from_activations(self, activations):
        return self.forward_head(self.features_from_activations(activations))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class InceptionNet3D(nn.Module):
    """Low-compute pure 3D Inception network (~3.1M parameters, 10x smaller than ResNet-18)
    with multi-scale feature extraction across all 4 stages and a 512-D TCAV bottleneck."""

    def __init__(
        self,
        num_classes=4,
        base_channels=64,
        dropout=0.5,
        use_residual=False,
        pooling="roi_attention",
    ):
        super().__init__()
        if (
            num_classes != 4
            or base_channels < 4
            or base_channels % 4 != 0
            or not (0.0 <= float(dropout) < 1.0)
        ):
            raise ValueError("Invalid InceptionNet3D settings")
        self.base_channels = base_channels
        self.shortcut = "A"
        self.dropout_rate = float(dropout)
        self.pooling_mode = pooling
        c = base_channels
        self.feature_dim = c * 8

        self.conv1 = nn.Conv3d(1, c, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = InceptionBlock3D(c, c, stride=1, use_residual=use_residual)
        self.layer2 = InceptionBlock3D(c, c * 2, stride=2, use_residual=use_residual)
        self.layer3 = InceptionBlock3D(c * 2, c * 4, stride=2, use_residual=use_residual)
        self.layer4 = InceptionBlock3D(c * 4, c * 8, stride=2, use_residual=use_residual)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.spatial_pool = _build_pooler(pooling, self.feature_dim)
        self.dropout = (
            nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        )
        self.fc = nn.Linear(self.feature_dim, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_layer4_activations(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def forward_features(self, x):
        return self.features_from_activations(self.get_layer4_activations(x))

    def features_from_activations(self, activations):
        if self.spatial_pool is not None:
            return self.spatial_pool(activations)
        return self.avgpool(activations).flatten(1)

    def forward_head(self, features):
        return self.fc(self.dropout(features))

    def forward_from_activations(self, activations):
        return self.forward_head(self.features_from_activations(activations))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


def build_model(config, input_shape=None):
    architecture = config.get("architecture", "ResNet18")
    dropout = float(config.get("dropout", 0.0))
    pooling = config.get("pooling", "gap")
    if architecture == "ResNet18":
        return ResNet18(
            config["num_classes"],
            config["base_channels"],
            config["shortcut"],
            dropout=dropout,
            pooling=pooling,
        )
    if architecture in {"InceptionResNet3D", "HybridInceptionResNet3D"}:
        return HybridInceptionResNet3D(
            config["num_classes"],
            config["base_channels"],
            config.get("shortcut", "A"),
            dropout=dropout,
            pooling=pooling,
        )
    if architecture == "Inception3D":
        return InceptionNet3D(
            config["num_classes"],
            config["base_channels"],
            dropout=dropout,
            use_residual=False,
            pooling=pooling,
        )
    raise ValueError(
        f"Unsupported model architecture: {architecture}. "
        "Supported: ResNet18, Inception3D, InceptionResNet3D, HybridInceptionResNet3D"
    )

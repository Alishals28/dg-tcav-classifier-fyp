"""3D MRI Classifier architectures for DG-TCAV and Yang et al. (PLOS ONE 2025) Replication:

1. Yang et al. (2025) 3D ResNet + CBAM Family (`Resnet-CBAM` branch):
   - `YangResNet18_CBAM`: 3D ResNet-18 with [2, 2, 2, 2] BasicBlocks, F=[32, 64, 128, 256], 1x1x1 downsample, 3D CBAM after Layer 4
   - `YangResNet34_CBAM`: 3D ResNet-34 with [3, 4, 6, 3] BasicBlocks, F=[32, 64, 128, 256], 1x1x1 downsample, 3D CBAM after Layer 4
   - `YangResNet50_CBAM`: 3D ResNet-50 with [3, 4, 6, 3] BottleneckBlocks (Fig 2a), F=[32, 64, 128, 256], 1x1x1 downsample, 3D CBAM after Layer 4

2. MedicalNet / Hybrid Family:
   - `ResNet18`: MedicalNet 3D ResNet-18 (base_channels=64, shortcut A/B, dilated layers 3 & 4)
   - `HybridInceptionResNet3D`: MedicalNet stem (`conv1`..`layer3`) + 3D Inception-ResNet `layer4`
   - `Inception3D`: Low-compute 3D Inception network

All architectures expose the unified DG-TCAV bottleneck interface (`get_layer4_activations`,
`forward_features`, `features_from_activations`, `forward_head`, `forward_from_activations`)
so Phase 9 live-gradient TCAV directional derivatives work seamlessly across all models.
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


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module per Yang et al. (PLOS ONE 2025, Page 6, Fig 2 & Table 2).

    Sequentially applies:
    1. 3D Channel Attention: global average pooling + global max pooling -> shared 1x1x1 MLP -> Sigmoid
    2. 3D Spatial Attention: channel-wise average + max projection -> 3x3x3 Conv3d + BatchNorm3d -> Sigmoid
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Channel Attention
        avg_pool = F.adaptive_avg_pool3d(x, 1)
        max_pool = F.adaptive_max_pool3d(x, 1)
        ch_attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * ch_attn

        # 2. Spatial Attention
        sp_avg = x.mean(dim=1, keepdim=True)
        sp_max = x.amax(dim=1, keepdim=True)
        sp_attn = self.spatial(torch.cat([sp_avg, sp_max], dim=1))
        return x * sp_attn


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


class YangBottleneck3D(nn.Module):
    """3D Bottleneck residual block per Figure 2(a) and Table 2 of Yang et al. (PLOS ONE 2025, Page 6-7):
    1x1x1 Conv3d(F) -> BN -> ReLU -> 3x3x3 Conv3d(F) -> BN -> ReLU -> 1x1x1 Conv3d(out_channels) -> BN + residual -> ReLU.

    Note on expansion:
    - Figure 2(a) shows `1x1x1 Conv3d, 4F` (expansion = 4, so Layer 4 outputs 4 * 256 = 1024 channels).
    - Table 2 lists `Layer 1 (F=32), Layer 2 (F=64), Layer 3 (F=128), Layer 4 (F=256)`.
    We support `expansion=4` (strictly matching Figure 2a) as well as `expansion=1` (if keeping output width 256).
    """

    def __init__(self, in_channels: int, planes: int, stride: int = 1, expansion: int = 4):
        super().__init__()
        out_channels = planes * expansion
        self.conv1 = nn.Conv3d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # Table 2 footnote: "1x1x1 convolutional kernel is used for downsampling"
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + residual)


class YangResNet3D_CBAM(nn.Module):
    """Authentic 3D ResNet-18 / 3D ResNet-34 / 3D ResNet-50 + CBAM per Table 2 of Yang et al. (2025).

    Key specifications from Yang et al. (PLOS ONE 2025, Page 6-7, Table 2 & Fig 2):
    - Stem: 7x7x7 Conv3d (stride 2, padding 3) -> BN -> ReLU -> 3x3x3 MaxPool3d (stride 2, padding 1)
    - Stage widths: F = [32, 64, 128, 256] (controlled by `base_channels=32`)
    - Stage strides: Layer 1 (stride 1), Layer 2 (stride 2), Layer 3 (stride 2), Layer 4 (stride 2)
    - Downsampling: 1x1x1 Conv3d + BN (`shortcut="B"`) in the first block of Layer 2, Layer 3, and Layer 4
    - CBAM: Placed after Layer 4 (`use_cbam=True`), before Average Pooling -> Fully Connected
    - TCAV Interface: `get_layer4_activations(x)` returns the post-CBAM spatial feature map `[B, C_out, D', H', W']`
      with autograd graph attached so Phase 9 DG-TCAV works without modification.
    """

    STAGE_CONFIGS = {
        "YangResNet18_CBAM": ("basic", [2, 2, 2, 2]),
        "YangResNet34_CBAM": ("basic", [3, 4, 6, 3]),
        "YangResNet50_CBAM": ("bottleneck", [3, 4, 6, 3]),
    }

    def __init__(
        self,
        variant: str = "YangResNet18_CBAM",
        num_classes: int = 4,
        base_channels: int = 32,
        dropout: float = 0.0,
        use_cbam: bool = True,
        cbam_reduction: int = 16,
        bottleneck_expansion: int = 4,
        pooling: str = "gap",
    ):
        super().__init__()
        if variant not in self.STAGE_CONFIGS:
            raise ValueError(f"Unsupported YangResNet variant: {variant}")
        if base_channels < 4 or not (0.0 <= float(dropout) < 1.0):
            raise ValueError("Invalid YangResNet3D_CBAM configuration")

        block_type, layers = self.STAGE_CONFIGS[variant]
        self.variant = variant
        self.base_channels = base_channels
        self.shortcut = "B"  # Table 2 footnote: 1x1x1 conv downsampling
        self.dropout_rate = float(dropout)
        self.use_cbam = bool(use_cbam)
        self.pooling_mode = pooling

        c = base_channels  # 32 in Yang et al. Table 2
        exp = bottleneck_expansion if block_type == "bottleneck" else 1
        self.feature_dim = c * 8 * exp  # 256 for ResNet-18/34; 1024 (or 256) for ResNet-50

        # Stem per Table 2
        self.conv1 = nn.Conv3d(1, c, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages with F = [32, 64, 128, 256] and stride = [1, 2, 2, 2]
        self.in_planes = c
        self.layer1 = self._make_stage(block_type, c, layers[0], stride=1, exp=exp)
        self.layer2 = self._make_stage(block_type, c * 2, layers[1], stride=2, exp=exp)
        self.layer3 = self._make_stage(block_type, c * 4, layers[2], stride=2, exp=exp)
        self.layer4 = self._make_stage(block_type, c * 8, layers[3], stride=2, exp=exp)

        # Post-Layer 4 3D CBAM per Table 2 and Figure 2
        self.cbam = (
            CBAM3D(self.feature_dim, reduction=cbam_reduction)
            if self.use_cbam
            else nn.Identity()
        )

        # Head per Table 2
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

    def _make_stage(self, block_type: str, planes: int, num_blocks: int, stride: int, exp: int):
        blocks = []
        if block_type == "basic":
            blocks.append(BasicBlock(self.in_planes, planes, stride=stride, dilation=1, shortcut="B"))
            self.in_planes = planes
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(self.in_planes, planes, stride=1, dilation=1, shortcut="B"))
        else:
            blocks.append(YangBottleneck3D(self.in_planes, planes, stride=stride, expansion=exp))
            self.in_planes = planes * exp
            for _ in range(1, num_blocks):
                blocks.append(YangBottleneck3D(self.in_planes, planes, stride=1, expansion=exp))
        return nn.Sequential(*blocks)

    def get_layer4_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return post-CBAM Layer 4 spatial feature maps with autograd graph attached for TCAV."""
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.cbam(x)

    def features_from_activations(self, activations: torch.Tensor) -> torch.Tensor:
        if self.spatial_pool is not None:
            return self.spatial_pool(activations)
        return self.avgpool(activations).flatten(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features_from_activations(self.get_layer4_activations(x))

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(features))

    def forward_from_activations(self, activations: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.features_from_activations(activations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


class InceptionBlock3D(nn.Module):
    """Multi-scale 3D Inception module with optional residual skip connection and 3D CBAM."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        use_residual=True,
        use_cbam=False,
        spatial_dropout=0.0,
    ):
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

        self.cbam = CBAM3D(out_channels) if use_cbam else nn.Identity()
        self.drop3d = (
            nn.Dropout3d(p=float(spatial_dropout))
            if float(spatial_dropout) > 0
            else nn.Identity()
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
        out = self.drop3d(self.cbam(out))
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
        _, _, d, h, w = shape
        z = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype).view(1, 1, d, 1, 1)
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, 1, h, 1)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, 1, w)
        prior = 0.5 + 0.5 * torch.exp(-((z / 0.65) ** 2 + (y / 0.65) ** 2 + (x / 0.70) ** 2))
        return prior

    def forward(self, activations):
        b, c, d, h, w = activations.shape
        logits = self.attn_conv(activations)
        if self.use_roi_prior:
            prior = self._medial_temporal_prior(activations.shape, activations.device, activations.dtype)
            logits = logits + torch.log(prior)
        weights = torch.softmax(logits.flatten(2), dim=-1).view(b, 1, d, h, w)
        attn_pooled = (activations * weights).sum(dim=(2, 3, 4))
        gap_pooled = activations.mean(dim=(2, 3, 4))
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
    """Four-class MedicalNet 3D ResNet-18 with configurable Dropout, CBAM, and Spatial Attention Pooling."""

    def __init__(
        self,
        num_classes=4,
        base_channels=64,
        shortcut="A",
        dropout=0.0,
        pooling="gap",
        use_cbam=False,
    ):
        super().__init__()
        if (
            shortcut not in {"A", "B"}
            or base_channels < 1
            or num_classes < 2
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
        self.cbam = CBAM3D(self.feature_dim) if use_cbam else nn.Identity()
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
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.cbam(self.layer4(self.layer3(self.layer2(self.layer1(x)))))

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


class HybridInceptionResNet3D(nn.Module):
    """Supervisor Hybrid Architecture: MedicalNet-compatible 3D ResNet stem (`conv1`..`layer3`)
    coupled with Multi-Scale 3D Inception-ResNet blocks in `layer4`."""

    def __init__(
        self,
        num_classes=4,
        base_channels=64,
        shortcut="A",
        dropout=0.5,
        pooling="roi_attention",
        use_cbam=False,
        spatial_dropout=0.0,
    ):
        super().__init__()
        if (
            shortcut not in {"A", "B"}
            or base_channels < 4
            or base_channels % 4 != 0
            or num_classes < 2
            or not (0.0 <= float(dropout) < 1.0)
        ):
            raise ValueError("Invalid HybridInceptionResNet3D settings")
        self.base_channels, self.shortcut = base_channels, shortcut
        self.dropout_rate = float(dropout)
        self.pooling_mode = pooling
        c = base_channels
        self.feature_dim = c * 8

        self.conv1 = nn.Conv3d(1, c, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(c)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._resnet_stage(c, c, 1, 1)
        self.layer2 = self._resnet_stage(c, c * 2, 2, 1)
        self.layer3 = self._resnet_stage(c * 2, c * 4, 1, 2)

        self.layer4 = nn.Sequential(
            InceptionBlock3D(
                c * 4,
                c * 8,
                stride=2,
                use_residual=True,
                use_cbam=use_cbam,
                spatial_dropout=spatial_dropout,
            ),
            InceptionBlock3D(
                c * 8,
                c * 8,
                stride=1,
                use_residual=True,
                use_cbam=use_cbam,
                spatial_dropout=spatial_dropout,
            ),
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
    """Low-compute pure 3D Inception network (~3.1M parameters)."""

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
            num_classes < 2
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
    spatial_dropout = float(config.get("spatial_dropout", 0.0))
    use_cbam = bool(config.get("use_cbam", False))
    if architecture in {"YangResNet18_CBAM", "YangResNet34_CBAM", "YangResNet50_CBAM"}:
        return YangResNet3D_CBAM(
            variant=architecture,
            num_classes=config["num_classes"],
            base_channels=config.get("base_channels", 32),
            dropout=dropout,
            use_cbam=config.get("use_cbam", True),
            cbam_reduction=int(config.get("cbam_reduction", 16)),
            bottleneck_expansion=int(config.get("bottleneck_expansion", 4)),
            pooling=pooling,
        )
    if architecture == "ResNet18":
        return ResNet18(
            config["num_classes"],
            config["base_channels"],
            config["shortcut"],
            dropout=dropout,
            pooling=pooling,
            use_cbam=use_cbam,
        )
    if architecture in {"InceptionResNet3D", "HybridInceptionResNet3D"}:
        return HybridInceptionResNet3D(
            config["num_classes"],
            config["base_channels"],
            config.get("shortcut", "A"),
            dropout=dropout,
            pooling=pooling,
            use_cbam=use_cbam,
            spatial_dropout=spatial_dropout,
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
        "Supported: ResNet18, Inception3D, InceptionResNet3D, HybridInceptionResNet3D, "
        "YangResNet18_CBAM, YangResNet34_CBAM, YangResNet50_CBAM"
    )

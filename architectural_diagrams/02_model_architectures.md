# Model Architecture Diagrams

This document details the primary neural network architecture implemented in [`src/model.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/model.py):
1. **MedicalNet 3D ResNet-18 Backbone** (Pretrained 3D Residual Network)
2. **Residual BasicBlock Sub-architecture** (Shortcut Type A)
3. **TCAV Concept Extraction Interface** (Layer-4 Activations & Bottleneck Features)

---

## 1. MedicalNet 3D ResNet-18 Architecture

The 3D ResNet-18 is a four-stage convolutional residual network adapted for 3D volumetric medical imaging. In transfer-learning profiles ([`configs/colab_medicalnet_resnet18_128.yaml`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/configs/colab_medicalnet_resnet18_128.yaml)), it loads backbone weights pre-trained across 23 3D medical datasets (Tencent Med3D). It uses `AdaptiveAvgPool3d(1)` to create a compact, spatially invariant 512-dimensional bottleneck.

```mermaid
flowchart TD
    IN["Input MRI Scan<br/>[Batch, 1, 128, 128, 128]"] --> CONV1
    
    subgraph STEM["Stem (Early Feature Extraction)"]
        CONV1["Conv3d(1, 64, kernel=7, stride=2, padding=3, bias=False)<br/>Output: [B, 64, 64, 64, 64]"]
        BN1["BatchNorm3d(64) + ReLU"]
        POOL1["MaxPool3d(kernel=3, stride=2, padding=1)<br/>Output: [B, 64, 32, 32, 32]"]
        CONV1 --> BN1 --> POOL1
    end

    subgraph STAGE1["Stage 1 (Layer 1) - 64 Channels"]
        BB1_1["BasicBlock 1.1 (stride=1, dilation=1)<br/>[B, 64, 32, 32, 32]"]
        BB1_2["BasicBlock 1.2 (stride=1, dilation=1)<br/>[B, 64, 32, 32, 32]"]
        BB1_1 --> BB1_2
    end

    subgraph STAGE2["Stage 2 (Layer 2) - 128 Channels"]
        BB2_1["BasicBlock 2.1 (stride=2, dilation=1)<br/>[B, 128, 16, 16, 16]"]
        BB2_2["BasicBlock 2.2 (stride=1, dilation=1)<br/>[B, 128, 16, 16, 16]"]
        BB2_1 --> BB2_2
    end

    subgraph STAGE3["Stage 3 (Layer 3) - 256 Channels (MedicalNet Dilated)"]
        BB3_1["BasicBlock 3.1 (stride=1, dilation=2)<br/>[B, 256, 16, 16, 16]"]
        BB3_2["BasicBlock 3.2 (stride=1, dilation=2)<br/>[B, 256, 16, 16, 16]"]
        BB3_1 --> BB3_2
    end

    subgraph STAGE4["Stage 4 (Layer 4) - 512 Channels (MedicalNet Dilated)"]
        BB4_1["BasicBlock 4.1 (stride=1, dilation=4)<br/>[B, 512, 16, 16, 16]"]
        BB4_2["BasicBlock 4.2 (stride=1, dilation=4)<br/>[B, 512, 16, 16, 16]"]
        BB4_1 --> BB4_2
    end

    POOL1 --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 --> STAGE3
    STAGE3 --> STAGE4

    STAGE4 -->|"get_layer4_activations(x)"| HOOK["Layer-4 Spatial Activations<br/>[Batch, 512, 16, 16, 16]<br/>(TCAV Extraction Point: get_layer4_activations)"]

    subgraph HEAD["Spatial Pooling & Classification Head"]
        HOOK --> AVG["AdaptiveAvgPool3d(output_size=1)<br/>Output: [Batch, 512, 1, 1, 1]"]
        AVG --> FLAT["Flatten(dim=1)<br/>Bottleneck Vector: [Batch, 512]"]
        FLAT --> FC["Linear(in_features=512, out_features=4)<br/>Weights: 2,048 | Biases: 4"]
        FC --> OUT["Logits: [Batch, 4]<br/>(CN: 0, EMCI: 1, LMCI: 2, AD: 3)"]
    end

    classDef stem fill:#e0f7fa,stroke:#00838f,stroke-width:2px;
    classDef stage fill:#f1f8e9,stroke:#558b2f,stroke-width:2px;
    classDef hook fill:#fff8e1,stroke:#f57f17,stroke-width:2px;
    classDef head fill:#ede7f6,stroke:#4527a0,stroke-width:2px;
    class STEM stem;
    class STAGE1,STAGE2,STAGE3,STAGE4 stage;
    class HOOK hook;
    class HEAD head;
```

---

## 2. Residual BasicBlock Sub-Architecture (Shortcut Type A)

Each stage consists of two residual BasicBlocks. When downsampling or increasing channels, MedicalNet uses parameter-free **Shortcut A** (`F.avg_pool3d` + zero padding), keeping the autograd graph clean and memory footprint light.

```mermaid
flowchart TD
    X["Input Tensor x<br/>[B, C_in, D, H, W]"] --> SPLIT{"Does stride != 1 or<br/>C_in != C_out?"}
    
    subgraph RESIDUAL["Residual Path (Shortcut A)"]
        SPLIT -->|Yes: Downsample / Expansion| POOL["F.avg_pool3d(kernel=1, stride=S)<br/>Subsamples spatial resolution"]
        POOL --> PAD["torch.cat([x, zeros], dim=1)<br/>Zero-pads remaining channels"]
        SPLIT -->|No: Identity| IDENT["Identity Mapping<br/>(Pass x unchanged)"]
    end

    subgraph MAIN["Main Convolutional Path"]
        X --> C1["Conv3d(C_in, C_out, kernel=3, stride=S, padding=D, dilation=D, bias=False)"]
        C1 --> BN1["BatchNorm3d(C_out)"]
        BN1 --> R1["ReLU()"]
        R1 --> C2["Conv3d(C_out, C_out, kernel=3, stride=1, padding=D, dilation=D, bias=False)"]
        C2 --> BN2["BatchNorm3d(C_out)"]
    end

    PAD & IDENT --> ADD((+))
    BN2 --> ADD
    ADD --> R2["ReLU()"]
    R2 --> Y["Output Tensor y<br/>[B, C_out, D/S, H/S, W/S]"]

    classDef main fill:#e8eaf6,stroke:#283593,stroke-width:2px;
    classDef res fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
    class C1,BN1,R1,C2,BN2 main;
    class POOL,PAD,IDENT res;
```

---

## 3. Layer Specifications & Parameter Count

| Layer / Stage | Submodules | Output Tensor Shape ($128^3$ Input) | Total Parameters |
|:---|:---|:---:|:---:|
| **Stem** | `Conv3d(1, 64, 7, s=2, p=3)` + `BN` + `MaxPool3d(3, s=2)` | `[Batch, 64, 32, 32, 32]` | **22,080** |
| **Stage 1 (Layer 1)** | 2 × BasicBlock (64 ch, stride=1, dilation=1) | `[Batch, 64, 32, 32, 32]` | **442,880** |
| **Stage 2 (Layer 2)** | 2 × BasicBlock (128 ch, stride=2, dilation=1) | `[Batch, 128, 16, 16, 16]` | **1,549,312** |
| **Stage 3 (Layer 3)** | 2 × BasicBlock (256 ch, stride=1, dilation=2) | `[Batch, 256, 16, 16, 16]` | **6,195,200** |
| **Stage 4 (Layer 4)** | 2 × BasicBlock (512 ch, stride=1, dilation=4) | `[Batch, 512, 16, 16, 16]` | **24,776,704** |
| **Global Pooling** | `AdaptiveAvgPool3d(1)` + `Flatten(1)` | `[Batch, 512]` | **0** |
| **Classification Head** | `Linear(512, 4)` | `[Batch, 4]` | **2,052** |
| **Total ResNet-18** | Full model with Shortcut Type A | `[Batch, 4]` | **32,988,228** |

---

## 4. TCAV Concept Extraction Interface

To support downstream Concept Activation Vector (CAV) extraction, [`src/model.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/model.py) exposes direct inspection methods on `ResNet18`:

```mermaid
classDiagram
    class ResNet18 {
        +base_channels: int = 64
        +shortcut: str = "A"
        +feature_dim: int = 512
        +conv1: Conv3d
        +bn1: BatchNorm3d
        +maxpool: MaxPool3d
        +layer1: Sequential
        +layer2: Sequential
        +layer3: Sequential
        +layer4: Sequential
        +avgpool: AdaptiveAvgPool3d(1)
        +fc: Linear(512, 4)
        +get_layer4_activations(x) [B, 512, 16, 16, 16]
        +features_from_activations(acts) [B, 512]
        +forward_head(feats) [B, 4]
        +forward_from_activations(acts) [B, 4]
        +forward(x) [B, 4]
    }
```

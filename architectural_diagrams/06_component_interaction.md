# Component Interaction & Hardware Pipeline Diagram

This document illustrates the internal component interactions that take place across CPU threads, GPU CUDA streams, and persistent disk storage during a single training batch step and across epoch boundaries.

---

## 1. Single Training Step Component Interaction Diagram

```mermaid
sequenceDiagram
    autonumber
    participant CPU_Worker as DataLoader Worker (CPU)
    participant Host_Mem as Pinned Host RAM
    participant CUDA_Stream as CUDA Stream / VRAM
    participant AMP as torch.autocast & GradScaler
    participant Backbone as 3D ResNet-18 Model
    participant Loss_Fn as CrossEntropyLoss
    participant Optim as AdamW Optimizer
    participant Monitor as tqdm Progress Monitor

    Note over CPU_Worker,Host_Mem: --- 1. Asynchronous Data Prefetching ---
    CPU_Worker->>CPU_Worker: Read NIfTI from SSD (nibabel)
    CPU_Worker->>CPU_Worker: Apply 3D Affine & Blur (TorchIO)
    CPU_Worker->>Host_Mem: Collate 4 volumes into batch tensor [4, 1, 128, 128, 128]
    
    Note over Host_Mem,CUDA_Stream: --- 2. Non-Blocking Device Transfer ---
    Host_Mem->>CUDA_Stream: images.to('cuda', non_blocking=True)
    Host_Mem->>CUDA_Stream: targets.to('cuda', non_blocking=True)
    
    Note over CUDA_Stream,Loss_Fn: --- 3. Forward Pass & Loss ---
    CUDA_Stream->>AMP: Enter autocast(device_type='cuda', dtype=torch.float16)
    AMP->>Backbone: Forward pass on 4D 3D volumes
    Backbone-->>AMP: Compute logits [4, 4]
    AMP->>Loss_Fn: Evaluate CrossEntropy(logits, targets)
    Loss_Fn-->>AMP: loss (scalar float tensor)
    
    Note over AMP,Optim: --- 4. Scaled Backward Pass & Gradient Clipping ---
    AMP->>AMP: scaler.scale(loss).backward()
    AMP->>Optim: scaler.unscale_(optimizer)
    AMP->>Backbone: torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    AMP->>Optim: scaler.step(optimizer) [Update weights]
    AMP->>AMP: scaler.update() [Adjust dynamic loss scale]
    
    Note over Loss_Fn,Monitor: --- 5. Telemetry & User Feedback ---
    Loss_Fn->>Monitor: Accumulate loss * weight_sum
    Monitor-->>Monitor: Update live ETA, speed (it/s), and running loss
```

---

## 2. Epoch Boundary Component Interaction & Checkpoint Safety

At the conclusion of each epoch, multiple monitoring, scheduling, and storage components coordinate to ensure that model weights, metrics, and random generator states are persisted via temporary-file replacement to minimize checkpoint corruption across epoch boundaries if Google Colab disconnects.

```mermaid
flowchart TD
    subgraph TRAIN_ENGINE["Training Engine (src/train.py)"]
        EPOCH_END["Training Epoch Complete<br/>(278 batches finished)"] --> VAL_START["Trigger Validation Pass<br/>(model.eval(), no_grad)"]
        VAL_START --> METRICS["Compute Metrics (src/metrics.py)<br/>- Accuracy<br/>- Macro-F1 (Selection Metric)<br/>- Balanced Accuracy<br/>- Per-Class Recall"]
        
        METRICS --> SCHED["Update LR Scheduler<br/>scheduler.step()<br/>(CosineAnnealingLR down to 1e-6)"]
        
        METRICS --> CMP{"Is val_macro_f1 > best?"}
    end

    subgraph STATE_CAPTURE["State Capture (Reproducibility Engine)"]
        CMP --> CAPTURE["Capture Complete State Snapshot:<br/>- model_state_dict<br/>- optimizer_state_dict<br/>- scheduler_state_dict<br/>- scaler_state_dict<br/>- RNG states (Python, PyTorch, NumPy, CUDA)<br/>- Full history list<br/>- bad_epochs & best_score"]
    end

    subgraph ATOMIC_IO["Atomic Checkpointing System (save_checkpoint)"]
        CAPTURE --> WRITE_TMP["Write Snapshot to Temporary File:<br/>checkpoint.tmp (Prevents corrupted half-written files)"]
        WRITE_TMP --> REPLACE["Atomic File Replacement:<br/>temporary.replace(final_path)"]
    end

    subgraph DRIVE["Google Drive Persistent Storage"]
        REPLACE --> LAST["Overwritten Epoch Boundary: last.pt<br/>(Enables resume from last completed epoch)"]
        
        CMP -->|Yes: Improved| SAVE_BEST["Save Best State: best.pt"]
        REPLACE --> SAVE_BEST
        CMP -->|Yes: Improved| VAL_PRED["Save: validation_predictions.csv<br/>validation_metrics.json"]
        
        REPLACE --> HIST_CSV["Full Rewrite Each Epoch: history.csv"]
        HIST_CSV --> PLOT["Generated Once at Run Completion:<br/>learning_curves.png"]
    end

    classDef engine fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef state fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    classDef atomic fill:#fffde7,stroke:#fbc02d,stroke-width:2px;
    classDef drive fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    class TRAIN_ENGINE engine;
    class STATE_CAPTURE state;
    class ATOMIC_IO atomic;
    class DRIVE drive;
```

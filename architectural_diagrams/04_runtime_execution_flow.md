# Runtime & Execution Flow Diagrams

This document illustrates the dynamic runtime behavior, state transitions, and step-by-step sequence of the training, validation, checkpointing, and evaluation lifecycle executed by [`src/train.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/train.py) and [`src/engine.py`](file:///e:/Desktop/git_repos/dg-tcav-classifier-fyp/src/engine.py).

---

## 1. Complete Training & Validation Sequence Diagram

> [!NOTE] Cohort and Batch Count Alignment
> The numbers depicted below reflect the active production profile (`configs/colab_medicalnet_resnet18_128.yaml`) running on the full preprocessed cohort:
> - **Audited Volumes:** 1,112 train + 236 validation = **1,348 scans** (matches live startup audit log)
> - **Training Batches:** $\lceil 1,112 / 4 \rceil = \mathbf{278\text{ batches}}$ per epoch at `batch_size: 4` (matches live tqdm progress bar `278/278`)
> - **Validation Batches:** $\lceil 236 / 4 \rceil = \mathbf{59\text{ batches}}$ per epoch at `batch_size: 4`
> 
> *(Note on alternative runs: The historical exploratory Kaggle run used `batch_size: 2`, and an early intermediate local subset contained only 535 train scans [134 batches]. The active Colab training run executes on the full verified 1,112-train / 236-val cohort).*

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Notebook
    participant Train as src.train (Driver)
    participant Config as src.config
    participant Audit as src.dataset (Audit)
    participant Model as src.model (ResNet18)
    participant MedNet as src.medicalnet
    participant Engine as src.engine (run_epoch)
    participant Storage as Google Drive Storage

    User->>Train: python -m src.train --config colab_medicalnet_resnet18_128.yaml
    Train->>Config: load_config() & require_preprocessing()
    Config-->>Train: Config verified & validated
    
    Train->>Audit: audit_dataset(index, splits=['train', 'val'])
    Audit-->>Train: Audit passed=True (1,348 scans verified)
    Train->>Storage: write_json("data_audit.json", "environment.json")
    
    Train->>Model: build_model(architecture="ResNet18", base_channels=64)
    Model-->>Train: Model instantiated on CPU
    
    opt If pretrained == True and resume is None
        Train->>MedNet: load_medicalnet(model, "resnet_18_23dataset.pth")
        MedNet->>Storage: Read pretrained weights (~126 MB)
        MedNet->>Model: Inject matching backbone weights (Strip 'module.', Skip 'fc.')
        MedNet->>Storage: write_json("medicalnet_load_report.json")
    end
    
    opt If resume is specified
        Train->>Storage: torch.load("last.pt")
        Train->>Model: load_state_dict(model_state_dict)
        Train->>Train: Restore optimizer, scheduler, scaler, RNG seeds, and bad_epochs
    end
    
    Train->>Model: model.to(device='cuda')
    
    loop Epoch = start + 1 to 100
        rect rgb(232, 245, 233)
            Note over Train,Engine: --- 1. Training Phase (model.train) ---
            Train->>Engine: run_epoch(model, train_loader, device, criterion, optimizer, scaler, clip_norm=1.0)
            loop For each batch in train_loader (278 batches of 4 scans)
                Engine->>Model: Forward pass under torch.autocast(fp16)
                Model-->>Engine: Logits [4, 4]
                Engine->>Engine: loss = criterion(logits, targets)
                Engine->>Model: scaler.scale(loss).backward()
                Engine->>Engine: scaler.unscale_(optimizer)
                Engine->>Model: clip_grad_norm_(parameters, max_norm=1.0)
                Engine->>Model: scaler.step(optimizer)
                Engine->>Engine: scaler.update()
                Engine->>User: tqdm progress bar update (batch ETA, running loss)
            end
            Engine-->>Train: train_metrics (loss, accuracy, macro_f1)
        end
        
        rect rgb(225, 245, 254)
            Note over Train,Engine: --- 2. Validation Phase (model.eval) ---
            Train->>Engine: run_epoch(model, val_loader, device, criterion)
            loop For each batch in val_loader (59 batches of 4 scans)
                Engine->>Model: Forward pass without gradient (torch.no_grad)
                Model-->>Engine: Logits [4, 4]
                Engine->>Engine: Accumulate confusion matrix & per-class predictions
            end
            Engine-->>Train: val_metrics (loss, accuracy, macro_f1, balanced_accuracy)
        end
        
        Train->>Train: scheduler.step() [CosineAnnealingLR]
        
        alt If val_macro_f1 > best
            Train->>Train: best = val_macro_f1, bad_epochs = 0
            Train->>Storage: save_checkpoint("best.pt", snapshot)
            Train->>Storage: write_csv("validation_predictions.csv")
            Train->>Storage: write_json("validation_metrics.json")
        else If val_macro_f1 <= best
            Train->>Train: bad_epochs += 1
        end
        
        Train->>Storage: save_checkpoint("last.pt", snapshot)
        Train->>Storage: write_csv("history.csv")
        Train->>User: print("Epoch {epoch}/100 loss={loss} val_macro_f1={f1} best={best}")
        
        opt If bad_epochs >= early_stopping_patience (15)
            Note over Train: Early stopping triggered! Peak performance reached.
            Train->>Storage: plot_history("learning_curves.png")
            Train-->>User: Training complete. Outputs saved to Drive.
        end
    end
```

---

## 2. Training Run Lifecycle State Diagram

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED

    UNINITIALIZED --> AUDITING : Start src.train
    AUDITING --> CONFIG_ERROR : Audit Failed (mismatched spacing / affine)
    CONFIG_ERROR --> [*] : Exit with ValueError

    AUDITING --> MODEL_INIT : Audit Passed (1,348 scans valid)
    
    state MODEL_INIT {
        [*] --> INSTANTIATE_BACKBONE
        INSTANTIATE_BACKBONE --> LOAD_PRETRAINED : pretrained=True
        LOAD_PRETRAINED --> MOVE_TO_CUDA : 100% backbone weights loaded
        INSTANTIATE_BACKBONE --> MOVE_TO_CUDA : pretrained=False
        MOVE_TO_CUDA --> RESTORE_RESUME : resume specified
        RESTORE_RESUME --> [*] : RNG & Epoch Restored
        MOVE_TO_CUDA --> [*] : Fresh Run
    }

    MODEL_INIT --> TRAINING_LOOP

    state TRAINING_LOOP {
        [*] --> TRAIN_EPOCH
        TRAIN_EPOCH --> VAL_EPOCH : 278 training batches complete
        VAL_EPOCH --> EVALUATE_METRIC : 59 validation batches complete
        
        EVALUATE_METRIC --> SAVE_BEST : val_macro_f1 > best
        SAVE_BEST --> SAVE_LAST : Save best.pt & metrics.json
        
        EVALUATE_METRIC --> INCREMENT_PATIENCE : val_macro_f1 <= best
        INCREMENT_PATIENCE --> SAVE_LAST : bad_epochs += 1
        
        SAVE_LAST --> CHECK_PATIENCE : Update last.pt & history.csv
        
        CHECK_PATIENCE --> TRAIN_EPOCH : bad_epochs < 15 and epoch < 100
    }

    TRAINING_LOOP --> EARLY_STOPPED : bad_epochs >= 15
    TRAINING_LOOP --> MAX_EPOCHS_REACHED : epoch == 100
    
    EARLY_STOPPED --> PLOT_AND_FINALIZE
    MAX_EPOCHS_REACHED --> PLOT_AND_FINALIZE

    PLOT_AND_FINALIZE --> FROZEN_TEST_EVAL : Trigger src.evaluate (--confirm-final-test)
    PLOT_AND_FINALIZE --> TCAV_ACTIVATION_EXPORT : Trigger src.export_activations
    
    FROZEN_TEST_EVAL --> [*]
    TCAV_ACTIVATION_EXPORT --> [*]
```

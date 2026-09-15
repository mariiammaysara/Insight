"""
Insight: DenseNet-121 Training Pipeline for Pediatric Pneumonia Classification.

Implements full training workflow aligned with docs/methodology.md:
- Stratified class-weighted CrossEntropyLoss
- AdamW optimizer with decoupled weight decay
- ReduceLROnPlateau dynamic learning rate scheduler
- Early stopping tracking validation loss
- Model checkpointing to outputs/checkpoints/best_model.pth
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for cross-platform Arabic/Unicode terminal support
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data.dataset import get_dataloaders
from src.model.model import build_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Executes one full training epoch over the training DataLoader.

    Args:
        model: DenseNet-121 PyTorch model.
        loader: Training DataLoader yielding (images, labels).
        criterion: Loss function (e.g. weighted CrossEntropyLoss).
        optimizer: Optimizer instance (e.g. AdamW).
        device: Device to execute computation on ('cuda' or 'cpu').

    Returns:
        Average training loss across all samples in the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return epoch_loss


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates model performance on any DataLoader without gradient tracking.

    Args:
        model: DenseNet-121 PyTorch model.
        loader: Evaluation DataLoader (validation or test).
        criterion: Loss function.
        device: Target compute device.

    Returns:
        Tuple of (average_loss, accuracy_fraction).
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def compute_class_weights(
    manifest_path: Union[str, Path] = "data/processed/manifest.csv",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Calculates class weights from the 'label' column for split='train' only.

    Follows the formula defined in docs/methodology.md:
        w_c = N / (C * N_c)
    where:
        N   = Total training samples
        C   = Number of distinct classes (2: Normal, Pneumonia)
        N_c = Sample count for class c

    Args:
        manifest_path: Path to manifest.csv.
        device: Optional device to transfer the weights tensor to.

    Returns:
        1D torch.Tensor of shape (C,) with computed class weights.
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found at: {manifest_file.resolve()}")

    df = pd.read_csv(manifest_file)
    train_df = df[df["split"] == "train"]

    if len(train_df) == 0:
        raise ValueError(f"No records found for split='train' in {manifest_file}")

    total_samples = len(train_df)  # N
    counts = train_df["label"].value_counts().to_dict()
    unique_classes = sorted(list(counts.keys()))
    num_classes = len(unique_classes)  # C

    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        if n_c == 0:
            raise ValueError(f"Class {c} has 0 samples in train split.")
        w_c = total_samples / (num_classes * n_c)
        weights.append(w_c)

    weights_tensor = torch.tensor(weights, dtype=torch.float)
    if device is not None:
        target_device = torch.device(device) if isinstance(device, str) else device
        weights_tensor = weights_tensor.to(target_device)

    return weights_tensor


if __name__ == "__main__":
    print("=" * 72)
    print("INSIGHT: DENSENET-121 TRAINING PIPELINE INITIALIZATION")
    print("=" * 72)

    # 1. Device detection & allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Selected compute device: {device}")
    print(f"[Device] Device type: {device.type}")
    print(f"[Device] CUDA available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[Device] GPU Hardware: {torch.cuda.get_device_name(0)}")
        print(f"[Device] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Paths
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"

    # 3. DataLoaders
    print("\n[Data] Loading datasets from manifest...")
    dataloaders = get_dataloaders(
        manifest_path=manifest_path,
        batch_size=32,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    print(f"[Data] Train samples: {len(train_loader.dataset):,} ({len(train_loader)} batches)")
    print(f"[Data] Val samples  : {len(val_loader.dataset):,} ({len(val_loader)} batches)")

    # 4. Class Weights & Loss Criterion
    class_weights = compute_class_weights(manifest_path, device=device)
    print(f"[Loss] Computed class weights (w_c = N / (C * N_c)): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5. Model Architecture
    print("\n[Model] Building DenseNet-121 backbone & applying freezing strategy...")
    model = build_model(device=device)

    # 6. Optimizer & Learning Rate Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # 7. Training Loop Hyperparameters
    max_epochs = 20
    early_stopping_patience = 5
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    print("\n" + "=" * 72)
    print(f"STARTING TRAINING (Max Epochs: {max_epochs}, Early Stopping Patience: {early_stopping_patience})")
    print("=" * 72)

    total_training_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate on validation split
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Step scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.1e} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            delta = best_val_loss - val_loss
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model to {best_model_path} (Val Loss improved by {delta:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  --> No improvement in Val Loss for {epochs_without_improvement}/{early_stopping_patience} epoch(s).")
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n[Early Stopping] Early stopping condition met at epoch {epoch}.")
                break

    total_training_time = time.time() - total_training_start
    total_epochs_executed = len(train_losses)

    # 8. Verification & Reporting
    print("\n" + "=" * 72)
    print("INSIGHT: FINAL VERIFICATION & TRAINING SUMMARY REPORT")
    print("=" * 72)

    # 1) Hardware execution check
    is_training_on_gpu = (device.type == "cuda")
    print(f"1) Hardware Execution Verification:")
    print(f"   - Active device.type == 'cuda' : {is_training_on_gpu}")
    print(f"   - Device in use                : {device}")
    if is_training_on_gpu:
        print(f"   - Dedicated GPU               : {torch.cuda.get_device_name(0)}")

    # 2) Total epochs executed
    print(f"\n2) Epochs Execution:")
    print(f"   - Total Epochs Executed        : {total_epochs_executed} (out of max {max_epochs})")
    print(f"   - Early Stopping Triggered     : {total_epochs_executed < max_epochs}")
    print(f"   - Total Training Duration      : {total_training_time / 60:.2f} minutes ({total_training_time:.1f}s)")

    # 3) Checkpoint file verification
    print(f"\n3) Best Model Checkpoint Verification:")
    if best_model_path.exists():
        file_size_mb = best_model_path.stat().st_size / (1024 * 1024)
        print(f"   - Checkpoint Path              : {best_model_path}")
        print(f"   - Exists on Disk               : True")
        print(f"   - File Size                    : {file_size_mb:.2f} MB")
    else:
        print(f"   - [ERROR]: Checkpoint file NOT found at {best_model_path}")

    # 4) Best Performance Metrics
    print(f"\n4) Optimal Performance Achieved:")
    print(f"   - Best Validation Epoch        : Epoch {best_epoch}")
    print(f"   - Best Validation Loss         : {best_val_loss:.4f}")
    print(f"   - Best Validation Accuracy     : {best_val_accuracy * 100:.2f}%")

    # 5) Train Loss Trend Verification
    print(f"\n5) Train Loss General Trend Check:")
    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    is_decreasing = final_loss < initial_loss
    print(f"   - Initial Train Loss (Epoch 1) : {initial_loss:.4f}")
    print(f"   - Final Train Loss (Epoch {total_epochs_executed}) : {final_loss:.4f}")
    print(f"   - Overall Trend Direction      : {'DECREASING (Convergence Confirmed)' if is_decreasing else 'NOT DECREASING (Warning!)'}")

    if not is_decreasing:
        print("   - [WARNING]: Train loss did not decrease overall. Please inspect learning rate and data.")
    else:
        print("   - [CONFIRMATION]: Model successfully converged and loss followed a healthy downward trajectory.")

    print("=" * 72 + "\n")

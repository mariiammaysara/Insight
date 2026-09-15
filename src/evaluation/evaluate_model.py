"""
Insight Evaluation: Held-Out Benchmark Test Set Evaluation Module.

Conducts rigorous, unbiased performance evaluation of the best DenseNet-121 checkpoint
on the strictly isolated official Kaggle test set (N=624) aligned with docs/methodology.md:
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, AUC-ROC
- Full Confusion Matrix plotting and artifact persistence
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless terminal plotting
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure project root is in sys.path
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


def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Iterates across all batches of the DataLoader, aggregates predictions, and computes
    clinical diagnostic metrics.

    Args:
        model: Trained DenseNet-121 PyTorch model.
        loader: DataLoader representing the target evaluation split.
        device: Device to run forward pass on ('cuda' or 'cpu').

    Returns:
        Dictionary containing metric values, confusion matrix, and prediction arrays.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            # Probability of positive class (PNEUMONIA = 1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Core metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "f1": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    save_path: Union[str, Path] = "outputs/reports/confusion_matrix.png",
    class_names: Optional[List[str]] = None,
) -> Path:
    """
    Renders and saves a high-resolution publication-quality confusion matrix plot.

    Args:
        cm: 2x2 confusion matrix array.
        save_path: Destination filepath for the PNG plot.
        class_names: List of display labels (default: ['NORMAL', 'PNEUMONIA']).

    Returns:
        Path to the saved image file.
    """
    if class_names is None:
        class_names = ["NORMAL", "PNEUMONIA"]

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d", colorbar=True)

    ax.set_title("Insight: Held-Out Benchmark Test Set (N=624)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=10, fontweight="bold")
    ax.set_ylabel("Ground Truth Pathology", fontsize=10, fontweight="bold")
    plt.tight_layout()

    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    return out_file


if __name__ == "__main__":
    print("=" * 74)
    print("INSIGHT: BENCHMARK HELD-OUT TEST EVALUATION (OFFICIAL TEST SPLIT)")
    print("=" * 74)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Evaluation compute device: {device}")

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"[Model] Loading best checkpoint from: {checkpoint_path}")
    model = build_model(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # Load official held-out test dataloader
    print("\n[Data] Initializing official_test DataLoader (624 benchmark scans)...")
    dataloaders = get_dataloaders(
        manifest_path=PROJECT_ROOT / "data" / "processed" / "manifest.csv",
        batch_size=32,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    official_loader = dataloaders["official_test"]
    print(f"[Data] Test samples loaded: {len(official_loader.dataset)} images across {len(official_loader)} batches.")

    # Execute full evaluation
    print("\n[Evaluation] Running full inference and metric calculations...")
    metrics = full_evaluation(model=model, loader=official_loader, device=device)

    # Plot and save confusion matrix
    cm_plot_path = PROJECT_ROOT / "outputs" / "reports" / "confusion_matrix.png"
    saved_plot = plot_and_save_confusion_matrix(metrics["confusion_matrix"], save_path=cm_plot_path)

    # Print Detailed Results
    print("\n" + "=" * 74)
    print("                   HELD-OUT BENCHMARK TEST RESULTS")
    print("=" * 74)
    print(f"1. Overall Accuracy       : {metrics['accuracy'] * 100:6.2f}% ({metrics['accuracy']:.4f})")
    print(f"2. Precision (Pneumonia)  : {metrics['precision'] * 100:6.2f}% ({metrics['precision']:.4f})")
    print(f"3. Recall / Sensitivity   : {metrics['recall'] * 100:6.2f}% ({metrics['recall']:.4f})")
    print(f"4. Specificity (Normal)   : {metrics['specificity'] * 100:6.2f}% ({metrics['specificity']:.4f})")
    print(f"5. F1-Score               : {metrics['f1'] * 100:6.2f}% ({metrics['f1']:.4f})")
    print(f"6. AUC-ROC                : {metrics['auc_roc'] * 100:6.2f}% ({metrics['auc_roc']:.4f})")
    print("-" * 74)
    print("Confusion Matrix Breakdown:")
    print(f"   - True Negatives  (TN - Normal correctly identified)    : {metrics['tn']}")
    print(f"   - False Positives (FP - Normal misclassified Pneumonia) : {metrics['fp']}")
    print(f"   - False Negatives (FN - Pneumonia missed / classified N): {metrics['fn']}")
    print(f"   - True Positives  (TP - Pneumonia correctly identified)  : {metrics['tp']}")
    print(f"   - Raw 2x2 Matrix:\n{metrics['confusion_matrix']}")
    print("=" * 74)

    # Mandatory Verification Checklist
    print("\n" + "=" * 74)
    print("INSIGHT: FINAL EVALUATION VERIFICATION CHECKLIST")
    print("=" * 74)

    # 1. Verification of 5 core numerical metrics
    print("1) Numerical Metrics Verification:")
    print(f"   * Accuracy   : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   * Precision  : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   * Recall     : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"   * F1-Score   : {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"   * AUC-ROC    : {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)")

    # 2. Confusion matrix sum check (must equal 624 exactly)
    cm_sum = int(metrics["confusion_matrix"].sum())
    expected_sum = 624
    sum_matches = (cm_sum == expected_sum)
    print(f"\n2) Confusion Matrix Total Sum Check:")
    print(f"   - Total evaluated scans : {cm_sum}")
    print(f"   - Expected total scans  : {expected_sum}")
    print(f"   - Exactly 624 evaluated : {sum_matches} (Zero skipped scans)")
    assert sum_matches, f"ERROR: Confusion matrix sum {cm_sum} != {expected_sum}!"

    # 3. Verification of confusion_matrix.png saved on disk
    print(f"\n3) Confusion Matrix Plot Verification:")
    if saved_plot.exists():
        plot_size_kb = saved_plot.stat().st_size / 1024.0
        print(f"   - Plot File Path        : {saved_plot.resolve()}")
        print(f"   - File Exists on Disk   : True")
        print(f"   - File Size             : {plot_size_kb:.2f} KB")
    else:
        print(f"   - [ERROR]: Plot file not found at {saved_plot}")
    assert saved_plot.exists(), "Confusion matrix image was not created!"

    print("=" * 74 + "\n")

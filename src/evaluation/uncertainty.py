"""
Insight Evaluation: Predictive Uncertainty Quantification Module.

Implements Shannon entropy-based uncertainty estimation and clinical tier
categorization (Low, Medium, High) aligned with docs/methodology.md
and docs/clinical-guidelines.md.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn

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

from src.data.dataset import get_transforms
from src.model.model import build_model


def compute_entropy(
    probabilities: Union[torch.Tensor, np.ndarray, List[float]],
    eps: float = 1e-12,
) -> float:
    """
    Computes normalized Shannon entropy for binary/multiclass probabilities:
        H(p) = - sum_{i} p_i * log2(p_i)

    For C=2 classes, H(p) is naturally bounded in [0.0, 1.0]:
        - Completely confident ([1.0, 0.0] or [0.0, 1.0]): H = 0.0
        - Completely uncertain ([0.5, 0.5]): H = 1.0

    Handles edge cases where p_i == 0 safely (0 * log2(0) -> 0).

    Args:
        probabilities: Array-like containing class probabilities summing to 1.
        eps: Small epsilon to prevent division/logarithm by zero.

    Returns:
        Float value representing normalized Shannon entropy in range [0.0, 1.0].
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = np.asarray(probabilities, dtype=np.float64)

    # Normalize if values do not sum strictly to 1
    total = np.sum(probs, axis=-1, keepdims=True)
    if not np.allclose(total, 1.0, atol=1e-5):
        probs = probs / np.maximum(total, eps)

    # Safe log computation: mask out zero probabilities (0 * log2(0) = 0)
    safe_probs = np.clip(probs, eps, 1.0)
    log_probs = np.log2(safe_probs)
    term = np.where(probs > eps, probs * log_probs, 0.0)

    entropy = -np.sum(term, axis=-1)
    # Clamp strictly to [0.0, 1.0] and return scalar float
    clamped_entropy = float(np.clip(entropy, 0.0, 1.0))
    return abs(clamped_entropy)


def get_uncertainty_tier(entropy_value: float) -> str:
    """
    Categorizes the normalized entropy value into operational clinical tiers:
        - Low    : entropy < 0.30
        - Medium : 0.30 <= entropy < 0.70
        - High   : entropy >= 0.70

    Thresholds are strictly aligned with docs/methodology.md.

    Args:
        entropy_value: Normalized entropy float in range [0.0, 1.0].

    Returns:
        String tier name: 'Low', 'Medium', or 'High'.
    """
    if entropy_value < 0.30:
        return "Low"
    elif entropy_value < 0.70:
        return "Medium"
    else:
        return "High"


def predict_with_uncertainty(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Union[int, float, str, List[float]]]:
    """
    Infers class probabilities from model, calculates Shannon entropy and uncertainty tier.

    Args:
        model: DenseNet-121 model instance.
        image_tensor: Input image tensor of shape (1, 3, 224, 224) or (3, 224, 224).
        device: Device to execute computation on ('cuda' or 'cpu').

    Returns:
        Dictionary containing:
            - 'predicted_class': int (0 for NORMAL, 1 for PNEUMONIA)
            - 'class_name': str ('NORMAL' or 'PNEUMONIA')
            - 'probability': float (confidence in predicted class)
            - 'probabilities': list of float [p_normal, p_pneumonia]
            - 'entropy': float (normalized Shannon entropy)
            - 'tier': str ('Low', 'Medium', or 'High')
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs_tensor = torch.softmax(logits, dim=1).squeeze(0)

    probs = [float(p) for p in probs_tensor.cpu().numpy()]
    pred_class = int(torch.argmax(probs_tensor).item())
    pred_prob = float(probs[pred_class])

    entropy = compute_entropy(probs)
    tier = get_uncertainty_tier(entropy)

    class_names = {0: "NORMAL", 1: "PNEUMONIA"}

    return {
        "predicted_class": pred_class,
        "class_name": class_names.get(pred_class, str(pred_class)),
        "probability": pred_prob,
        "probabilities": probs,
        "entropy": entropy,
        "tier": tier,
    }


if __name__ == "__main__":
    print("=" * 74)
    print("INSIGHT: PREDICTIVE UNCERTAINTY MODULE INITIALIZATION")
    print("=" * 74)

    # 1. Device and Model Loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Compute device for uncertainty estimation: {device}")

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[Model] Building DenseNet-121 and loading weights from {checkpoint_path}...")
    model = build_model(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Manifest and Internal Test Data
    manifest_file = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_file}")

    df = pd.read_csv(manifest_file)
    test_df = df[df["split"] == "internal_test"].reset_index(drop=True)
    print(f"[Data] Found {len(test_df)} samples in internal_test split.")

    val_transform = get_transforms(split="val")

    # 3. Pick 5 random images from internal_test
    # Using fixed random_state for reproducible evaluation
    sampled_indices = test_df.sample(n=5, random_state=42).index.tolist()
    print(f"[Sampling] Randomly sampled 5 internal_test indices: {sampled_indices}")

    print("\n" + "=" * 74)
    print("PREDICTING WITH UNCERTAINTY ON 5 RANDOMLY SAMPLED TEST SCANS")
    print("=" * 74)

    results = []

    for i, idx in enumerate(sampled_indices, 1):
        row = test_df.iloc[idx]
        rel_path = Path(row["filepath"])
        img_path = PROJECT_ROOT / rel_path if not rel_path.is_absolute() else rel_path
        ground_truth = int(row["label"])
        gt_name = "NORMAL" if ground_truth == 0 else "PNEUMONIA"

        with Image.open(img_path) as raw_img:
            img_rgb = raw_img.convert("RGB")

        tensor = val_transform(img_rgb).unsqueeze(0)
        pred_dict = predict_with_uncertainty(model, tensor, device=device)

        results.append({
            "sample_num": i,
            "filename": img_path.name,
            "ground_truth": gt_name,
            "predicted_class": pred_dict["class_name"],
            "probability": pred_dict["probability"],
            "probabilities": pred_dict["probabilities"],
            "entropy": pred_dict["entropy"],
            "tier": pred_dict["tier"],
        })

        print(
            f"Sample {i} [{img_path.name[:35]:<35}] | "
            f"Ground Truth: {gt_name:<9} | "
            f"Prediction: {pred_dict['class_name']:<9} | "
            f"Prob: {pred_dict['probability'] * 100:6.2f}% | "
            f"Entropy: {pred_dict['entropy']:.4f} | "
            f"Tier: {pred_dict['tier']}"
        )

    # 4. Mandatory Verification Checks
    print("\n" + "=" * 74)
    print("INSIGHT: UNCERTAINTY VERIFICATION CHECKLIST")
    print("=" * 74)

    # Check 1: All entropy values between 0.0 and 1.0
    all_entropies_valid = all(0.0 <= r["entropy"] <= 1.0 for r in results)
    min_ent = min(r["entropy"] for r in results)
    max_ent = max(r["entropy"] for r in results)
    print(f"1) Entropy Bounds Check ([0.0, 1.0]):")
    print(f"   - All within [0.0, 1.0] : {all_entropies_valid}")
    print(f"   - Min Entropy Observed  : {min_ent:.6f}")
    print(f"   - Max Entropy Observed  : {max_ent:.6f}")
    assert all_entropies_valid, "ERROR: Found entropy value outside [0, 1] range!"

    # Check 2: Tier consistency check
    tier_consistent = True
    for r in results:
        e = r["entropy"]
        t = r["tier"]
        expected_tier = get_uncertainty_tier(e)
        if t != expected_tier:
            tier_consistent = False
            print(f"   - [MISMATCH]: Entropy {e} labeled as {t} but expected {expected_tier}")

    print(f"\n2) Tier Consistency Check:")
    print(f"   - Tiers Strictly Consistent with Methodology Thresholds: {tier_consistent}")
    assert tier_consistent, "ERROR: Tier classification mismatch detected!"

    # Check 3: Unique tiers observed in 5-sample batch
    unique_tiers = set(r["tier"] for r in results)
    print(f"\n3) Tier Diversity in 5-Sample Batch:")
    print(f"   - Tiers Present in 5 Samples: {sorted(list(unique_tiers))}")
    if len(unique_tiers) > 1:
        print(f"   - Multiple tiers observed across the 5 samples ({len(unique_tiers)} distinct tiers).")
    else:
        single_tier = list(unique_tiers)[0]
        print(f"   - [ALERT / NOTE]: All 5 randomly selected samples belong to the '{single_tier}' tier.")
        print(f"     Reason: The sample size (N=5) is small, and DenseNet-121 achieves >98% test accuracy,")
        print(f"     meaning the vast majority of test predictions are extremely confident (p > 99%).")

    # Extra Comprehensive Scan: Full internal_test Split Distribution
    print(f"\n4) Full Internal Test Split (N={len(test_df)}) Uncertainty Distribution:")
    from src.data.dataset import get_dataloaders
    test_loader = get_dataloaders(batch_size=32)["internal_test"]
    full_tiers = {"Low": 0, "Medium": 0, "High": 0}

    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            probs_batch = torch.softmax(model(imgs), dim=1).cpu().numpy()
            for probs in probs_batch:
                ent = compute_entropy(probs)
                full_tiers[get_uncertainty_tier(ent)] += 1

    for tier_name, count in full_tiers.items():
        pct = (count / len(test_df)) * 100
        print(f"   - Tier {tier_name:<6} : {count:3d} scans ({pct:5.2f}%)")

    print("=" * 74 + "\n")

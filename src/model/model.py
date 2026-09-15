"""
DenseNet-121 Model Architecture for Pneumonia Detection (Insight).

Implements DenseNet-121 initialized with ImageNet pre-trained weights, modifies
the classification head for binary prediction, and applies a two-stage freezing
strategy aligned with docs/methodology.md (CheXNet literature grounding).
"""

import sys
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

# Ensure UTF-8 output encoding for cross-platform Arabic/Unicode terminal support
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_densenet121(pretrained: bool = True) -> nn.Module:
    """
    Constructs a DenseNet-121 backbone and replaces the classification head.

    Uses modern torchvision weights API (DenseNet121_Weights.IMAGENET1K_V1).
    The original 1000-class linear classifier is replaced with:
        nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=2)
        )

    Args:
        pretrained: If True, downloads/loads ImageNet pre-trained weights.

    Returns:
        Modified DenseNet-121 nn.Module.
    """
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)

    in_features = model.classifier.in_features  # 1024 for DenseNet-121

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=in_features, out_features=2),
    )

    return model


def freeze_early_layers(model: nn.Module) -> nn.Module:
    """
    Applies the parameter freezing strategy documented in methodology.md.

    Frozen (requires_grad = False):
        - features.conv0, features.norm0
        - features.denseblock1, features.transition1
        - features.denseblock2, features.transition2
        - features.denseblock3, features.transition3

    Trainable (requires_grad = True):
        - features.denseblock4 (high-level domain-specific pulmonary patterns)
        - features.norm5 (final feature normalization)
        - classifier (new binary head)

    Args:
        model: DenseNet-121 model instance.

    Returns:
        Model with selectively frozen parameters.
    """
    layers_to_freeze = [
        model.features.conv0,
        model.features.norm0,
        model.features.denseblock1,
        model.features.transition1,
        model.features.denseblock2,
        model.features.transition2,
        model.features.denseblock3,
        model.features.transition3,
    ]

    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False

    # Ensure deep adaptation layers remain strictly trainable
    layers_to_train = [
        model.features.denseblock4,
        model.features.norm5,
        model.classifier,
    ]

    for layer in layers_to_train:
        for param in layer.parameters():
            param.requires_grad = True

    # Parameter accounting
    trainable_params, total_params, pct_trainable = count_parameters(model)
    print("\n" + "-" * 60)
    print("INSIGHT: DENSENET-121 PARAMETER FREEZING SUMMARY")
    print("-" * 60)
    print(f"Total Parameters      : {total_params:,}")
    print(f"Trainable Parameters  : {trainable_params:,}")
    print(f"Frozen Parameters     : {total_params - trainable_params:,}")
    print(f"Trainable Ratio       : {pct_trainable:.2f}%")
    print("-" * 60 + "\n")

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Counts trainable and total parameters in the model.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (trainable_params, total_params, percentage_trainable).
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct_trainable = (trainable_params / total_params * 100) if total_params > 0 else 0.0
    return trainable_params, total_params, pct_trainable


def build_model(device: Optional[Union[str, torch.device]] = None) -> nn.Module:
    """
    Full factory pipeline: builds DenseNet-121, applies parameter freezing,
    and transfers the model to the target device.

    Args:
        device: Target torch device ('cuda', 'cpu', or torch.device).

    Returns:
        Configured and device-allocated PyTorch model ready for training/inference.
    """
    model = build_densenet121(pretrained=True)
    model = freeze_early_layers(model)

    if device is not None:
        device = torch.device(device) if isinstance(device, str) else device
        model = model.to(device)

    return model


if __name__ == "__main__":
    print("=" * 68)
    print("RUNNING SANITY CHECK: DenseNet-121 Model Architecture")
    print("=" * 68)

    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device Selection] Target device chosen : {target_device}")

    model = build_model(device=target_device)

    # Verify device allocation
    actual_device = next(model.parameters()).device
    print(f"[Device Verification] Actual model device : {actual_device}")

    # Forward pass verification with dummy batch
    dummy_input = torch.randn(1, 3, 224, 224, device=target_device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"[Forward Pass] Dummy input shape       : {tuple(dummy_input.shape)}")
    print(f"[Forward Pass] Model output shape      : {tuple(output.shape)} (Expected: (1, 2))")
    print(f"[Forward Pass] Raw logits output       : {output.squeeze().tolist()}")

    # Parameter accounting check
    trainable, total, pct = count_parameters(model)
    print(f"[Parameter Check] Trainable parameters : {trainable:,} / {total:,} ({pct:.2f}%)")

    assert output.shape == torch.Size([1, 2]), f"Expected output shape (1, 2), got {output.shape}"
    assert not torch.isnan(output).any(), "Output logits contain NaN!"

    print("=" * 68)
    print("SANITY CHECK COMPLETED SUCCESSFULLY: Model architecture verified!")
    print("=" * 68)

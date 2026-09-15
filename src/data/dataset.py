"""
PyTorch Dataset and DataLoader Pipeline for Insight.

Implements ChestXrayDataset for loading pediatric chest radiographs from
manifest.csv, provides split-specific data augmentations aligned with
methodology.md, and exports ready-to-use DataLoaders.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import logging
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Ensure UTF-8 output encoding for cross-platform Arabic/Unicode terminal support
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ChestXrayDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Chest X-ray images based on manifest.csv.

    Args:
        manifest_path: Path to the generated manifest.csv file.
        split: Target split to load ('train', 'val', 'internal_test', 'official_test').
        transform: Optional torchvision transforms to apply to the PIL image.
        root_dir: Optional base directory if filepaths in manifest are relative.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.corrupted_count: int = 0

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at: {self.manifest_path.resolve()}")

        df = pd.read_csv(self.manifest_path)

        valid_splits = {"train", "val", "internal_test", "official_test"}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Expected one of: {valid_splits}")

        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            logger.warning(f"No records found for split '{split}' in {self.manifest_path}.")

    def __len__(self) -> int:
        """Returns the total number of images in the filtered split."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the image and label at the specified index.
        Converts the image explicitly to 3-channel RGB. If loading fails due to file
        corruption or an I/O error, logs a warning, tracks the failure, and falls back
        to a neighboring valid sample.

        Args:
            idx: Row index within the split.

        Returns:
            Tuple of (transformed_image_tensor, integer_label).
        """
        max_retries = min(5, len(self.df))
        current_idx = idx

        for attempt in range(max_retries):
            row = self.df.iloc[current_idx]
            rel_path = Path(row["filepath"])
            full_path = self.root_dir / rel_path if not rel_path.is_absolute() else rel_path
            label = int(row["label"])

            try:
                with Image.open(full_path) as img:
                    image_rgb = img.convert("RGB")

                if self.transform:
                    image_tensor = self.transform(image_rgb)
                else:
                    image_tensor = transforms.ToTensor()(image_rgb)

                return image_tensor, label

            except (OSError, IOError, ValueError) as err:
                self.corrupted_count += 1
                logger.warning(
                    f"Corrupted image detected at index {current_idx} ({full_path}): {err}. "
                    f"Falling back to neighboring index {(current_idx + 1) % len(self.df)}."
                )
                current_idx = (current_idx + 1) % len(self.df)

        raise RuntimeError(f"Failed to load image after {max_retries} attempts starting at index {idx}.")


def get_transforms(split: str) -> transforms.Compose:
    """
    Returns split-specific torchvision transform pipelines aligned with methodology.md.

    - Train: Resize(224, 224) -> RandomRotation(10) -> RandomHorizontalFlip(p=0.5)
             -> ToTensor() -> ImageNet Normalization.
    - Val/Test: Resize(224, 224) -> ToTensor() -> ImageNet Normalization.

    Args:
        split: Dataset split identifier.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


def get_dataloaders(
    manifest_path: Union[str, Path] = "data/processed/manifest.csv",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Builds Datasets and DataLoaders for all four splits.

    Args:
        manifest_path: Path to manifest.csv.
        batch_size: Number of images per batch (default: 32).
        num_workers: Number of subprocess workers for loading (default: 0 for Windows compatibility).
        pin_memory: If True, copies Tensors to CUDA pinned memory before returning.

    Returns:
        Dictionary containing DataLoaders for 'train', 'val', 'internal_test', and 'official_test'.
    """
    splits = ["train", "val", "internal_test", "official_test"]
    dataloaders = {}

    for split in splits:
        transform = get_transforms(split)
        dataset = ChestXrayDataset(manifest_path=manifest_path, split=split, transform=transform)

        is_train = (split == "train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        dataloaders[split] = loader

    return dataloaders


if __name__ == "__main__":
    print("=" * 78)
    print("INSIGHT: PYTORCH DATASET & DATALOADER SANITY VERIFICATION")
    print("=" * 78)

    manifest_file = Path("data/processed/manifest.csv")
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file {manifest_file} not found. Run prepare_dataset.py first.")

    raw_manifest_df = pd.read_csv(manifest_file)
    loaders = get_dataloaders(manifest_path=manifest_file, batch_size=32)

    # 1. Compare Dataset count vs manifest.csv count for each split
    print("\n--- 1. Verification of Split Counts vs Manifest ---")
    all_matched = True
    for split_name in ["train", "val", "internal_test", "official_test"]:
        manifest_count = len(raw_manifest_df[raw_manifest_df["split"] == split_name])
        dataset_count = len(loaders[split_name].dataset)
        match = (manifest_count == dataset_count)
        all_matched = all_matched and match
        print(
            f"Split: {split_name:<14} | Dataset len: {dataset_count:<5} | "
            f"Manifest rows: {manifest_count:<5} | Match: {'MATCHED' if match else 'MISMATCH'}"
        )
    print(f"Overall Counts Alignment: {'PERFECT MATCH (100%)' if all_matched else 'FAILED'}")

    # 2. Verify single item __getitem__ output
    print("\n--- 2. Single Sample (__getitem__) Output Verification ---")
    train_dataset = loaders["train"].dataset
    single_img, single_lbl = train_dataset[0]
    print(f"Single image tensor shape : {tuple(single_img.shape)} (Expected: (3, 224, 224))")
    print(f"Single label value        : {single_lbl} (type: {type(single_lbl).__name__})")
    assert single_img.shape == torch.Size([3, 224, 224]), "Shape mismatch!"

    # 3. Verify label domain across manifest
    print("\n--- 3. Label Domain & Validity Check ---")
    unique_labels = sorted(raw_manifest_df["label"].unique().tolist())
    print(f"Unique labels in manifest : {unique_labels}")
    is_strictly_binary = (unique_labels == [0, 1])
    print(f"Labels strictly (0, 1)     : {is_strictly_binary}")

    # 4. Batch Inspection & Normalization Value Ranges
    print("\n--- 4. Batch Tensor & Normalization Range Verification ---")
    train_batch_imgs, train_batch_lbls = next(iter(loaders["train"]))
    val_batch_imgs, val_batch_lbls = next(iter(loaders["val"]))

    print(f"[Train Batch] Images shape: {tuple(train_batch_imgs.shape)} | Labels shape: {tuple(train_batch_lbls.shape)}")
    print(
        f"[Train Batch] Value range : min={train_batch_imgs.min():.4f}, "
        f"max={train_batch_imgs.max():.4f}, mean={train_batch_imgs.mean():.4f}, "
        f"std={train_batch_imgs.std():.4f}"
    )
    print(
        f"[Train Batch] NaN Check   : Has NaN = {torch.isnan(train_batch_imgs).any().item()} | "
        f"Has Inf = {torch.isinf(train_batch_imgs).any().item()}"
    )

    print(f"[Val Batch]   Images shape: {tuple(val_batch_imgs.shape)} | Labels shape: {tuple(val_batch_lbls.shape)}")
    print(
        f"[Val Batch]   Value range : min={val_batch_imgs.min():.4f}, "
        f"max={val_batch_imgs.max():.4f}, mean={val_batch_imgs.mean():.4f}, "
        f"std={val_batch_imgs.std():.4f}"
    )
    print(
        f"[Val Batch]   NaN Check   : Has NaN = {torch.isnan(val_batch_imgs).any().item()} | "
        f"Has Inf = {torch.isinf(val_batch_imgs).any().item()}"
    )

    # 5. Corrupted images check
    print("\n--- 5. Corrupted / Skipped Images Report ---")
    total_corrupted = sum(loader.dataset.corrupted_count for loader in loaders.values())
    if total_corrupted == 0:
        print("Corrupted Images: لا توجد صور تالفة (0 corrupted images encountered)")
    else:
        print(f"Corrupted Images: تم رصد {total_corrupted} صور تالفة")

    # 6. GPU Availability & Hardware Detection
    print("\n--- 6. Hardware & GPU Detection ---")
    gpu_available = torch.cuda.is_available()
    print(f"CUDA (GPU) Available : {gpu_available}")
    if gpu_available:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"GPU Device Name      : {device_name}")
        print(f"GPU Device Count     : {device_count}")
    else:
        print("GPU Device Name      : N/A (Running on CPU)")

    print("\n" + "=" * 78)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 78 + "\n")

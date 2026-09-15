"""
Dataset Preparation & Stratified Splitting for Chest X-Ray Pneumonia Dataset.

This module scans the raw dataset directory, aggregates the original Kaggle
train and validation sets into a unified development pool, performs a stratified
re-split (70% train / 15% val / 15% internal_test), and preserves the official
test set as an isolated held-out benchmark. Outputs a reproducible manifest.csv.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def scan_dataset_directory(raw_dir: Path) -> List[Dict[str, any]]:
    """
    Scans the raw dataset directory structure and collects all image records.

    Expected directory structure:
        raw_dir/
            train/
                NORMAL/
                PNEUMONIA/
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/

    Args:
        raw_dir: Path to the raw dataset root (e.g. data/raw/chest_xray).

    Returns:
        A list of dictionaries containing 'filepath', 'label', and 'source_split'.
    """
    valid_extensions = {".jpeg", ".jpg", ".png", ".dcm"}
    label_map = {"NORMAL": 0, "PNEUMONIA": 1}
    records = []

    for split_dir in ["train", "val", "test"]:
        split_path = raw_dir / split_dir
        if not split_path.exists():
            continue

        for class_name, label_idx in label_map.items():
            class_path = split_path / class_name
            if not class_path.exists():
                continue

            for file_path in class_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                    # Store path relative to project root with forward slashes
                    try:
                        rel_path = file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
                    except ValueError:
                        rel_path = file_path.as_posix()

                    records.append({
                        "filepath": rel_path,
                        "label": label_idx,
                        "source_split": split_dir
                    })

    return records


def perform_stratified_split(
    records: List[Dict[str, any]],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Merges original train and val splits into a development pool, applies a
    70/15/15 stratified split, and preserves the official test split.

    Args:
        records: List of scanned image dictionaries.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns ['filepath', 'label', 'split'].
    """
    df_all = pd.DataFrame(records)
    if df_all.empty:
        raise ValueError("No images found in the specified raw directory.")

    # Separate development pool (train + val) from official test set
    dev_mask = df_all["source_split"].isin(["train", "val"])
    test_mask = df_all["source_split"] == "test"

    df_dev = df_all[dev_mask].copy()
    df_official_test = df_all[test_mask].copy()

    # Step 1: 70% train, 30% temp (which will become 15% val + 15% internal_test)
    train_df, temp_df = train_test_split(
        df_dev,
        test_size=0.30,
        stratify=df_dev["label"],
        random_state=random_state
    )

    # Step 2: Split the 30% temp equally into val (15%) and internal_test (15%)
    val_df, internal_test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=random_state
    )

    # Assign target split names
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    internal_test_df = internal_test_df.assign(split="internal_test")
    df_official_test = df_official_test.assign(split="official_test")

    # Combine into unified manifest DataFrame
    manifest_df = pd.concat(
        [train_df, val_df, internal_test_df, df_official_test],
        ignore_index=True
    )

    # Keep strictly specified schema: filepath, label, split
    manifest_df = manifest_df[["filepath", "label", "split"]]
    return manifest_df


def print_statistical_summary(df: pd.DataFrame) -> None:
    """
    Prints a clear terminal summary of sample counts, class distributions,
    and imbalance ratios across all splits.

    Args:
        df: Processed manifest DataFrame.
    """
    split_order = ["train", "val", "internal_test", "official_test"]
    class_names = {0: "Normal", 1: "Pneumonia"}

    print("\n" + "=" * 78)
    print("INSIGHT: DATASET SPLIT & CLASS DISTRIBUTION SUMMARY")
    print("=" * 78)
    print(f"{'Split':<16} | {'Total':<7} | {'Normal (0)':<12} | {'Pneumonia (1)':<14} | {'Imbalance Ratio':<15}")
    print("-" * 78)

    for split_name in split_order:
        sub = df[df["split"] == split_name]
        if sub.empty:
            continue

        total = len(sub)
        counts = sub["label"].value_counts().to_dict()
        n_normal = counts.get(0, 0)
        n_pneu = counts.get(1, 0)

        pct_normal = (n_normal / total * 100) if total > 0 else 0
        pct_pneu = (n_pneu / total * 100) if total > 0 else 0
        ratio_str = f"{n_pneu / n_normal:.2f} : 1" if n_normal > 0 else "N/A"

        norm_str = f"{n_normal} ({pct_normal:.1f}%)"
        pneu_str = f"{n_pneu} ({pct_pneu:.1f}%)"

        print(f"{split_name:<16} | {total:<7} | {norm_str:<12} | {pneu_str:<14} | {ratio_str:<15}")

    print("-" * 78)
    total_samples = len(df)
    total_normal = (df["label"] == 0).sum()
    total_pneu = (df["label"] == 1).sum()
    overall_ratio = f"{total_pneu / total_normal:.2f} : 1" if total_normal > 0 else "N/A"

    print(
        f"{'TOTAL':<16} | {total_samples:<7} | "
        f"{total_normal} ({total_normal / total_samples * 100:.1f}%) | "
        f"{total_pneu} ({total_pneu / total_samples * 100:.1f}%) | "
        f"{overall_ratio:<15}"
    )
    print("=" * 78 + "\n")


def main() -> None:
    """Entry point for dataset preparation."""
    raw_dir = Path("data/raw/chest_xray")
    output_dir = Path("data/processed")
    output_file = output_dir / "manifest.csv"

    print(f"Scanning raw dataset in: {raw_dir.resolve()} ...")

    if not raw_dir.exists():
        print(f"\n[ERROR] Raw data directory not found: {raw_dir}")
        print("Please ensure the Kaggle 'chest_xray' dataset is extracted to 'data/raw/chest_xray/'.")
        print("Expected structure: data/raw/chest_xray/{train, val, test}/{NORMAL, PNEUMONIA}/\n")
        return

    records = scan_dataset_directory(raw_dir)
    print(f"Found {len(records)} image files across all splits.")

    if not records:
        print("[ERROR] No images discovered. Verify directory structure and image file extensions.")
        return

    manifest_df = perform_stratified_split(records, random_state=42)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_file, index=False)
    print(f"Manifest successfully generated and saved to: {output_file.as_posix()}")

    print_statistical_summary(manifest_df)


if __name__ == "__main__":
    main()

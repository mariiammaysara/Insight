"""
Insight Trust Report: Integrated Clinical Explainability & Uncertainty Decision Module.

Combines DenseNet-121 prediction, Grad-CAM anatomical heatmaps, and Shannon entropy
predictive uncertainty to generate a unified, actionable clinical Trust Report
aligned with docs/methodology.md and docs/clinical-guidelines.md.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
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
from src.evaluation.uncertainty import get_uncertainty_tier, predict_with_uncertainty
from src.explainability.gradcam import generate_gradcam, overlay_heatmap
from src.model.model import build_model

# Clinical operational decision sentences strictly defined in methodology.md
CLINICAL_DECISIONS: Dict[str, str] = {
    "Low": "يُوثق في هذه النتيجة — مراجعة قياسية كافية",
    "Medium": "يُنصح بمراجعة ثانوية والتحقق من الأعراض السريرية",
    "High": "مطلوب مراجعة إلزامية من أخصائي أشعة أول قبل اعتماد النتيجة",
}


def format_report_text(report_data: Dict[str, Any]) -> str:
    """Formats the trust report dictionary into a readable clinical text document."""
    lines = [
        "=" * 68,
        "             INSIGHT: CLINICAL TRUST & DECISION REPORT",
        "=" * 68,
        f"1. Image Identification : {report_data['filename']}",
        f"   - Full Input Path     : {report_data['input_path']}",
        "",
        "2. Diagnostic Prediction :",
        f"   - Predicted Diagnosis : {report_data['prediction']}",
        f"   - Prediction Index    : {report_data['predicted_class']}",
        f"   - Confidence (Prob)   : {report_data['probability'] * 100:.2f}%",
        f"   - Class Probabilities : Normal: {report_data['probabilities'][0] * 100:.2f}% | Pneumonia: {report_data['probabilities'][1] * 100:.2f}%",
        "",
        "3. Predictive Uncertainty :",
        f"   - Normalized Entropy  : {report_data['entropy']:.4f} (scale 0.0 - 1.0)",
        f"   - Operational Tier    : {report_data['tier']}",
        "",
        "4. Actionable Clinical Decision :",
        f"   >> \"{report_data['decision_sentence']}\"",
        "",
        "5. Visual Explainability (Grad-CAM) :",
        f"   - Grad-CAM Overlay    : {report_data['gradcam_overlay_path']}",
        f"   - Target Conv Layer   : {report_data['target_layer']}",
        "=" * 68,
    ]
    return "\n".join(lines)


def generate_trust_report(
    model: nn.Module,
    image_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    save_dir: Union[str, Path] = "outputs/reports",
) -> Dict[str, Any]:
    """
    Generates an integrated Trust Report for a chest radiograph.

    Steps:
    1. Loads and preprocesses the image using the validation/testing pipeline.
    2. Runs forward inference to obtain class probabilities, Shannon entropy, and uncertainty tier.
    3. Synthesizes a Grad-CAM heatmap targeting the final convolutional layer.
    4. Blends heatmap and radiograph into an explanatory visual overlay.
    5. Formulates clinical decision guidance mapped directly to the uncertainty tier.
    6. Saves overlay image and report artifacts (JSON and formatted text).

    Args:
        model: DenseNet-121 model.
        image_path: Path to target chest radiograph image.
        device: Device to execute computation on ('cuda' or 'cpu').
        save_dir: Destination directory for report artifacts.

    Returns:
        Dictionary containing all structured report elements.
    """
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Input image not found: {img_file.resolve()}")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Load image and apply transform
    val_transform = get_transforms(split="val")
    with Image.open(img_file) as raw_img:
        original_pil = raw_img.convert("RGB")

    tensor = val_transform(original_pil).unsqueeze(0).to(device)

    # 2. Uncertainty & Prediction
    pred_info = predict_with_uncertainty(model=model, image_tensor=tensor, device=device)
    tier = pred_info["tier"]
    decision_sentence = CLINICAL_DECISIONS[tier]

    # 3. Grad-CAM generation & overlay
    target_layer = model.features.denseblock4.denselayer16.conv2
    heatmap = generate_gradcam(
        model=model,
        image_tensor=tensor,
        target_layer=target_layer,
        device=device,
    )
    overlay_img = overlay_heatmap(original_pil, heatmap, alpha=0.5)

    # 4. Save artifacts
    stem = img_file.stem
    gradcam_file = save_path / f"{stem}_gradcam.png"
    json_file = save_path / f"{stem}_report.json"
    txt_file = save_path / f"{stem}_report.txt"

    overlay_img.save(gradcam_file, format="PNG")

    report_dict: Dict[str, Any] = {
        "filename": img_file.name,
        "input_path": str(img_file.resolve()),
        "prediction": pred_info["class_name"],
        "predicted_class": pred_info["predicted_class"],
        "probability": pred_info["probability"],
        "probabilities": pred_info["probabilities"],
        "entropy": pred_info["entropy"],
        "tier": tier,
        "decision_sentence": decision_sentence,
        "gradcam_overlay_path": str(gradcam_file.resolve()),
        "json_report_path": str(json_file.resolve()),
        "text_report_path": str(txt_file.resolve()),
        "target_layer": "features.denseblock4.denselayer16.conv2",
    }

    # Save JSON report
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(report_dict, jf, indent=2, ensure_ascii=False)

    # Save Formatted Text report
    formatted_text = format_report_text(report_dict)
    with open(txt_file, "w", encoding="utf-8") as tf:
        tf.write(formatted_text)

    report_dict["formatted_text"] = formatted_text
    return report_dict


if __name__ == "__main__":
    print("=" * 72)
    print("INSIGHT: TRUST REPORT SYSTEM INITIALIZATION")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running Trust Report generator on: {device}")

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model = build_model(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # Load test manifest to select 3 diverse cases representing Low, Medium, and High uncertainty
    manifest_file = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_file}")

    df = pd.read_csv(manifest_file)
    test_df = df[df["split"] == "internal_test"].reset_index(drop=True)
    val_transform = get_transforms("val")

    print("[Scanning] Selecting 3 representative cases (Low, Medium, High tiers) from internal_test...")
    selected_samples: Dict[str, Path] = {}

    for idx, row in test_df.iterrows():
        p = PROJECT_ROOT / row["filepath"]
        with Image.open(p) as img:
            t = val_transform(img.convert("RGB")).unsqueeze(0).to(device)
        res = predict_with_uncertainty(model, t, device=device)
        t_name = res["tier"]
        if t_name not in selected_samples:
            selected_samples[t_name] = p
        if len(selected_samples) == 3:
            break

    print(f"[Selected Samples] Found samples for tiers: {list(selected_samples.keys())}")

    reports = []
    output_reports_dir = PROJECT_ROOT / "outputs" / "reports"

    print("\n" + "=" * 72)
    print("GENERATING TRUST REPORTS FOR SELECTED TEST RADIOGRAPHS")
    print("=" * 72)

    for tier_target in ["Low", "Medium", "High"]:
        sample_path = selected_samples.get(tier_target)
        if sample_path is None:
            continue

        print(f"\n---> Generating Trust Report for [{tier_target} Uncertainty Case]: {sample_path.name}")
        rep = generate_trust_report(
            model=model,
            image_path=sample_path,
            device=device,
            save_dir=output_reports_dir,
        )
        reports.append(rep)

        # Print the complete formatted report to terminal
        print("\n" + rep["formatted_text"])

    # Mandatory Verification Checklist
    print("\n" + "=" * 72)
    print("INSIGHT: TRUST REPORT VERIFICATION CHECKLIST")
    print("=" * 72)

    # 1. Verification of 4 core elements (non-None, non-empty)
    core_keys = ["prediction", "probability", "tier", "decision_sentence"]
    all_elements_present = True
    for r in reports:
        for k in core_keys:
            val = r.get(k)
            if val is None or (isinstance(val, str) and not val.strip()):
                all_elements_present = False
                print(f"   - [ERROR]: Key '{k}' is missing or empty in report for {r['filename']}")

    print(f"1) Core Elements Completeness Check:")
    print(f"   - All 4 elements present and non-empty in all reports : {all_elements_present}")
    for r in reports:
        print(f"   * {r['filename']:<32} | Pred: {r['prediction']:<9} | Prob: {r['probability']*100:5.2f}% | Tier: {r['tier']:<6} | Decision: \"{r['decision_sentence'][:30]}...\"")
    assert all_elements_present, "Missing core element in reports!"

    # 2. Verification of saved artifacts on disk
    print(f"\n2) Saved Artifacts on Disk ({output_reports_dir}):")
    all_files_exist = True
    for r in reports:
        g_path = Path(r["gradcam_overlay_path"])
        j_path = Path(r["json_report_path"])
        t_path = Path(r["text_report_path"])
        for p in [g_path, j_path, t_path]:
            if p.exists():
                size_kb = p.stat().st_size / 1024.0
                print(f"   - [EXISTS] {p.name:<40} ({size_kb:6.2f} KB) -> {p}")
            else:
                all_files_exist = False
                print(f"   - [MISSING]: {p}")
    assert all_files_exist, "One or more report artifact files were not saved!"

    # 3. Exact matching between Tier and Decision Sentence
    print(f"\n3) Tier to Decision Sentence Consistency Check:")
    sentences_match_strictly = True
    for r in reports:
        actual_tier = r["tier"]
        actual_sentence = r["decision_sentence"]
        expected_sentence = CLINICAL_DECISIONS[actual_tier]
        matches = (actual_sentence == expected_sentence)
        if not matches:
            sentences_match_strictly = False
            print(f"   - [MISMATCH] in {r['filename']}: Tier '{actual_tier}' has sentence '{actual_sentence}' != '{expected_sentence}'")
        else:
            print(f"   - [VERIFIED] Tier '{actual_tier}' strictly matched expected clinical decision.")

    assert sentences_match_strictly, "Decision sentence mismatch detected!"
    print(f"   - All report sentences strictly aligned with operational protocol: {sentences_match_strictly}")

    print("=" * 72 + "\n")

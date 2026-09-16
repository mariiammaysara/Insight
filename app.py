"""
Insight: Pediatric Screening Console.
High-density, modern Clinical Imaging Workstation (PACS / Tactical Diagnostic Console style).
"""

from pathlib import Path
import sys
from PIL import Image
import torch
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model import build_model
from src.data.dataset import get_transforms
from src.evaluation.uncertainty import predict_with_uncertainty
from src.explainability.gradcam import generate_gradcam, overlay_heatmap
from src.evaluation.trust_report import CLINICAL_DECISIONS, CLINICAL_DECISIONS_EN


st.set_page_config(
    page_title="Insight — Clinical Screening",
    page_icon="assets/favicon.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# High-Density PACS Console CSS & Typography
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Apply globally across all Streamlit text elements, cards, and labels */
    html, body, [class*="css"], .stApp, .stMarkdown, .stButton, button, p, span, div, h1, h2, h3, h4, h5, h6 {
        font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace !important;
        letter-spacing: -0.01em;
    }

    html, body, .stApp {
        background-color: #080c14 !important;
        color: #f8fafc !important;
    }

    /* Hide Streamlit default chrome & headers */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {visibility: hidden; display: none;}
    div[data-testid="stDecoration"] {visibility: hidden; display: none;}
    div[data-testid="stStatusWidget"] {visibility: hidden; display: none;}
    section[data-testid="stSidebar"] {display: none;}

    /* Tighten layout margins */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1440px !important;
    }

    /* Viewport Headers */
    .viewport-header {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-bottom: none;
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #94a3b8;
    }

    /* Fixed-height image viewports to prevent layout shift */
    div[data-testid="stImage"] {
        height: 440px !important;
        max-height: 440px !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stImage"] img {
        height: 440px !important;
        max-height: 440px !important;
        width: 100% !important;
        object-fit: contain !important;
        background-color: #080c14 !important;
        border: 1px solid rgba(148, 163, 184, 0.12) !important;
        border-radius: 0 0 10px 10px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        box-sizing: border-box !important;
        display: block !important;
        margin: 0 !important;
    }

    /* Minimalist Wireframe Placeholder matching image viewport height */
    .wireframe-placeholder {
        width: 100% !important;
        height: 440px !important;
        min-height: 440px !important;
        max-height: 440px !important;
        border: 1px dashed rgba(148, 163, 184, 0.15) !important;
        border-radius: 0 0 10px 10px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        background: radial-gradient(circle at center, rgba(56, 189, 248, 0.03) 0%, transparent 70%) !important;
        color: #64748b !important;
        font-size: 11px !important;
        letter-spacing: 0.02em !important;
        text-align: center !important;
        padding: 24px !important;
        box-sizing: border-box !important;
        margin: 0 !important;
    }

    .pulse-radar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.1);
    }

    /* Triage & Telemetry Card */
    .triage-panel {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 18px 24px;
        margin-top: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }

    .triage-grid {
        display: grid;
        grid-template-columns: 1.4fr 1.2fr 1fr 1.6fr;
        gap: 16px;
        align-items: center;
    }

    .dock-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    /* Buttons Styling */
    .stButton > button {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        width: 100% !important;
        background: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        line-height: 1 !important;
        padding: 0 14px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
        box-sizing: border-box !important;
    }

    .stButton > button:hover {
        background: #334155 !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }

    /* File uploader compact styling aligned with dock buttons */
    div[data-testid="stFileUploader"] {
        padding: 0 !important;
        margin: 0 !important;
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        width: 100% !important;
    }
    div[data-testid="stFileUploader"] > section {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        width: 100% !important;
        background: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        line-height: 1 !important;
        padding: 0 14px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: #38bdf8 !important;
        background: #334155 !important;
        color: #ffffff !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }
    /* Hide cluttered raw default children */
    div[data-testid="stFileUploader"] section > * {
        display: none !important;
    }
    div[data-testid="stFileUploader"] ul {
        display: none !important;
    }
    /* Prepend clean SVG Upload Icon */
    div[data-testid="stFileUploader"] > section::before {
        content: "" !important;
        display: inline-block !important;
        width: 15px !important;
        height: 15px !important;
        min-width: 15px !important;
        min-height: 15px !important;
        margin-right: 6px !important;
        vertical-align: -2px !important;
        background-color: currentColor !important;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") no-repeat center !important;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") no-repeat center !important;
        -webkit-mask-size: contain !important;
        mask-size: contain !important;
        transition: all 0.2s ease !important;
    }
    /* Clean single label matching button text */
    div[data-testid="stFileUploader"] > section::after {
        content: "Upload Radiograph" !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        line-height: 1 !important;
        color: inherit !important;
        display: inline-block !important;
        white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Deep Learning Inference & Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def get_inference_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device=device)
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"

    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.eval()

    val_transform = get_transforms(split="val")
    return model, val_transform, device


model, val_transform, device = get_inference_pipeline()

# -----------------------------------------------------------------------------
# State & Fixture Configurations
# -----------------------------------------------------------------------------
if "active_image" not in st.session_state:
    st.session_state.active_image = None
    st.session_state.active_title = None
    st.session_state.diagnostic_results = None

sample_base = PROJECT_ROOT / "data" / "raw" / "chest_xray"
FIXTURES = {
    "normal": {
        "tag": "NORMAL_131",
        "label": "Normal Baseline",
        "path": sample_base / "test" / "NORMAL" / "NORMAL2-IM-0131-0001.jpeg",
        "meta": "Held-out Test #131 • Clear Parenchyma",
    },
    "pneumonia": {
        "tag": "PNEUMONIA_TP",
        "label": "Pneumonia Case",
        "path": sample_base / "test" / "PNEUMONIA" / "person100_bacteria_475.jpeg",
        "meta": "Held-out Test • Right Lobe Infiltrate",
    },
    "equivocal": {
        "tag": "HIGH_ENTROPY",
        "label": "Equivocal Edge Case",
        "path": sample_base / "train" / "PNEUMONIA" / "person635_bacteria_2526.jpeg",
        "meta": "Borderline Cohort • Epistemic Ambiguity",
    },
}


def process_radiograph(img: Image.Image, name: str):
    """Executes forward pass, Grad-CAM generation, and entropy quantification."""
    tensor = val_transform(img).unsqueeze(0).to(device)
    pred_info = predict_with_uncertainty(model=model, image_tensor=tensor, device=device)

    target_layer = model.features.denseblock4.denselayer16.conv2
    heatmap = generate_gradcam(
        model=model,
        image_tensor=tensor,
        target_layer=target_layer,
        device=device,
    )
    overlay = overlay_heatmap(img, heatmap, alpha=0.5)

    return {
        "pred_info": pred_info,
        "overlay": overlay,
        "name": name,
    }


# -----------------------------------------------------------------------------
# 1. Centered Header Block
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 24px;">
      <h1 style="font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; color: #f8fafc; margin: 0 0 4px 0;">Insight</h1>
      <div style="font-size: 13px; font-weight: 500; letter-spacing: 0.04em; color: #94a3b8;">
        Pediatric Radiograph Screening Console
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. Center Diagnostic Canvas (Split-View Radiograph Viewer)
# -----------------------------------------------------------------------------
col_raw, col_cam = st.columns(2, gap="medium")

results = st.session_state.diagnostic_results
active_img = st.session_state.active_image

with col_raw:
    st.markdown(
        """
        <div class="viewport-header">
          <span>Input Radiograph</span>
          <span style="font-size: 10px; color: #64748b;">Anterior-Posterior View</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if active_img is not None:
        st.image(active_img, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="wireframe-placeholder">
              <div class="pulse-radar"><span style="color:#38bdf8; font-size:16px;">+</span></div>
              <div style="color: #94a3b8; font-weight: 600; margin-bottom: 4px;">No Radiograph Selected</div>
              <div>Select a scenario or upload a chest radiograph to begin.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_cam:
    st.markdown(
        """
        <div class="viewport-header">
          <span>Grad-CAM Saliency Map</span>
          <span style="font-size: 10px; color: #38bdf8;">DenseBlock4 Layer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if results is not None:
        st.image(results["overlay"], use_container_width=True)
    else:
        st.markdown(
            """
            <div class="wireframe-placeholder">
              <div class="pulse-radar"><span style="color:#64748b; font-size:16px;">*</span></div>
              <div style="color: #94a3b8; font-weight: 600; margin-bottom: 4px;">Awaiting Analysis</div>
              <div>Saliency heatmaps will generate once a radiograph is selected.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# 3. Triage & Uncertainty Panel
# -----------------------------------------------------------------------------
if results is not None:
    pred_data = results["pred_info"]
    pred_label = pred_data["class_name"]
    prob = pred_data["probability"] * 100.0
    entropy = float(pred_data["entropy"])
    tier = pred_data["tier"]

    # Actionable Clinical Recommendation from src.evaluation.trust_report
    rec_text = CLINICAL_DECISIONS_EN.get(tier, CLINICAL_DECISIONS_EN["Low"])

    # Tier Badging
    tier_config = {
        "Low": {
            "badge": "Low Uncertainty",
            "bg": "rgba(16, 185, 129, 0.12)",
            "border": "#10b981",
            "text": "#34d399",
        },
        "Medium": {
            "badge": "Medium Uncertainty",
            "bg": "rgba(245, 158, 11, 0.12)",
            "border": "#f59e0b",
            "text": "#fbbf24",
        },
        "High": {
            "badge": "High Uncertainty",
            "bg": "rgba(245, 158, 11, 0.14)",
            "border": "#f59e0b",
            "text": "#fbbf24",
        },
    }
    cfg = tier_config.get(tier, tier_config["Medium"])

    # Dynamic Alert Styling: HIGH uncertainty overrides to Amber/Orange safety warning
    if tier == "High":
        triage_style = "background: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.4); box-shadow: 0 4px 20px rgba(245, 158, 11, 0.1);"
        condition_text = "Pneumonia Detected" if pred_label == "PNEUMONIA" else "Normal Clearance"
        condition_color = "#ef4444" if pred_label == "PNEUMONIA" else "#fbbf24"
        bar_color = "#ef4444" if pred_label == "PNEUMONIA" else "#fbbf24"
    elif pred_label == "PNEUMONIA":
        triage_style = "background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.35); box-shadow: 0 4px 20px rgba(239, 68, 68, 0.1);"
        condition_text = "Pneumonia Detected"
        condition_color = "#ef4444"
        bar_color = "#ef4444"
    else:
        triage_style = "background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.35); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);"
        condition_text = "Normal Clearance"
        condition_color = "#34d399"
        bar_color = "#38bdf8"

    triage_html = f"""<div class="triage-panel" style="{triage_style}"><div class="triage-grid">
<div>
<div style="font-size: 10px; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase;">Predicted Condition</div>
<div style="font-size: 16px; font-weight: 700; color: {condition_color}; margin-top: 4px; letter-spacing: 0.02em;">{condition_text}</div>
<div style="font-size: 10px; color: #64748b; margin-top: 2px;">File: {results['name']}</div>
</div>
<div>
<div style="font-size: 10px; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase;">Model Confidence</div>
<div style="font-size: 15px; font-weight: 700; color: #f8fafc; margin-top: 4px;">{prob:.2f}%</div>
<div style="background: rgba(255,255,255,0.06); height: 5px; border-radius: 2px; margin-top: 6px; overflow:hidden;">
<div style="background: {bar_color}; width: {prob}%; height: 100%;"></div>
</div>
</div>
<div>
<div style="font-size: 10px; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase;">Shannon Entropy</div>
<div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">
<span style="font-size: 15px; font-weight: 700; color: #38bdf8;">{entropy:.4f}</span>
<span style="padding: 2px 8px; font-size: 10px; font-weight: 600; border-radius: 3px; background: {cfg['bg']}; border: 1px solid {cfg['border']}; color: {cfg['text']};">{cfg['badge']}</span>
</div>
</div>
<div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 4px; padding: 10px 14px;">
<div style="font-size: 9px; font-weight: 600; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase;">Clinical Recommendation</div>
<div style="font-size: 0.85rem; font-weight: 600; color: #f8fafc; margin-top: 4px; line-height: 1.4;">{rec_text}</div>
</div>
</div></div>"""
    st.markdown(triage_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Quick-Input Dock (Bottom Control Strip)
# -----------------------------------------------------------------------------
st.markdown('<div class="dock-label">Instant Scenarios & Custom Upload</div>', unsafe_allow_html=True)

dock_c1, dock_c2, dock_c3, dock_c4 = st.columns(4, gap="small")

with dock_c1:
    if st.button(FIXTURES["normal"]["label"], use_container_width=True):
        p = FIXTURES["normal"]["path"]
        if p.exists():
            img = Image.open(p).convert("RGB")
            st.session_state.active_image = img
            st.session_state.active_title = FIXTURES["normal"]["tag"]
            st.session_state.diagnostic_results = process_radiograph(img, p.name)
            st.rerun()

with dock_c2:
    if st.button(FIXTURES["pneumonia"]["label"], use_container_width=True):
        p = FIXTURES["pneumonia"]["path"]
        if p.exists():
            img = Image.open(p).convert("RGB")
            st.session_state.active_image = img
            st.session_state.active_title = FIXTURES["pneumonia"]["tag"]
            st.session_state.diagnostic_results = process_radiograph(img, p.name)
            st.rerun()

with dock_c3:
    if st.button(FIXTURES["equivocal"]["label"], use_container_width=True):
        p = FIXTURES["equivocal"]["path"]
        if p.exists():
            img = Image.open(p).convert("RGB")
            st.session_state.active_image = img
            st.session_state.active_title = FIXTURES["equivocal"]["tag"]
            st.session_state.diagnostic_results = process_radiograph(img, p.name)
            st.rerun()

with dock_c4:
    custom_file = st.file_uploader(
        "Upload Custom Radiograph",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Upload an AP pediatric chest radiograph",
    )
    if custom_file is not None:
        img = Image.open(custom_file).convert("RGB")
        if st.session_state.active_title != custom_file.name:
            st.session_state.active_image = img
            st.session_state.active_title = custom_file.name
            st.session_state.diagnostic_results = process_radiograph(img, custom_file.name)
            st.rerun()

# -----------------------------------------------------------------------------
# 5. Minimal Product Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 28px; padding-top: 14px; border-top: 1px solid rgba(148, 163, 184, 0.08); text-align: center; font-size: 11px; color: #64748b; letter-spacing: 0.02em;">
      Insight • Pediatric Radiograph Screening Console • <a href="https://github.com/mariiammaysara/Insight" target="_blank" style="color: #94a3b8; text-decoration: none; font-weight: 500;">GitHub ↗</a>
    </div>
    """,
    unsafe_allow_html=True,
)

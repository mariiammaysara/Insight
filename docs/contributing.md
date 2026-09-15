<div align="center">

# Contributing Guidelines & Engineering Standards
### Insight: Pediatric Pneumonia Detection & Clinical Decision Support System

[![Philosophy](https://img.shields.io/badge/Philosophy-Simplicity%20First-blue?style=flat-square)](#1-code-philosophy-simplicity-first)
[![Architecture](https://img.shields.io/badge/Design-Separation%20of%20Concerns-brightgreen?style=flat-square)](#2-structural-integrity--separation-of-concerns)
[![Clinical](https://img.shields.io/badge/Ethical-Clinician--in--the--Loop-red?style=flat-square)](#7-ethical--clinical-safety-constraints)

[Code Philosophy](#1-code-philosophy-simplicity-first) • [Structural Integrity](#2-structural-integrity--separation-of-concerns) • [Documentation Standards](#3-documentation-standards) • [Code Quality](#4-code-quality--typing-standards) • [Pre-Commit Verification](#5-pre-commit-verification--sanity-checks) • [Commit Conventions](#6-commit-message-conventions) • [Clinical Safety](#7-ethical--clinical-safety-constraints)

</div>

---

## 1. Code Philosophy: Simplicity First

Insight is built on a clear software engineering ethos: **As simple as possible, without over-engineering.**

In machine learning and clinical decision support systems, overly complex abstractions, unnecessary metaprogramming, and bloated object hierarchies obscure bugs and compromise auditability. Contributors should:
- Prefer clean, readable, procedural pipelines over heavy inheritance trees.
- Choose explicit functions with transparent parameter contracts over opaque configuration objects.
- Avoid introducing external dependencies when native standard library utilities or standard PyTorch idioms suffice.

---

## 2. Structural Integrity & Separation of Concerns

Any new contribution or feature must strictly respect the existing **Separation of Concerns (SoC)** defined in [`docs/architecture.md`](architecture.md):

```text
src/
├── data/           # Strictly data ingestion, manifest parsing, augmentations, DataLoaders
├── model/          # Strictly model definition, layer freezing, optimization, training loops
├── explainability/ # Strictly saliency maps, Grad-CAM hooks, anatomical attention overlays
└── evaluation/     # Strictly uncertainty estimation, Trust Report orchestration, offline metrics
```

### Architectural Rules:
- **No Cross-Cutting Sprawl**: Do not place data transformation logic inside `src/model/` or compute evaluation metrics inside `src/explainability/`.
- **Modularity & Decoupling**: Functions in `src/evaluation/uncertainty.py` and `src/explainability/gradcam.py` must remain decoupled from specific training scripts, allowing them to operate on arbitrary model checkpoints.

---

## 3. Documentation Standards

Code without documentation is considered incomplete. Every addition must satisfy two levels of documentation:

### 1. In-Code Docstrings (Function & Class Level)
Every public function, method, and class must feature a Google/Sphinx-style docstring:
```python
def compute_metric(data: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes a brief summary sentence of what this function achieves.

    Detailed explanation of edge cases, assumptions, and mathematical context
    if applicable.

    Args:
        data: Description of input tensor or array dimensions and range.
        threshold: Decision boundary threshold value.

    Returns:
        Calculated metric float value bounded in [0.0, 1.0].

    Raises:
        ValueError: If data array is empty or contains NaN values.
    """
```

### 2. Architectural & Technical Documentation
Any non-trivial engineering decision (e.g., altering loss weights, changing a target convolutional layer, introducing a new uncertainty formula, or adjusting data splits) **must be documented in the relevant file in `docs/`**:
- Mathematical formulations $\rightarrow$ [`docs/methodology.md`](methodology.md)
- Architectural pipelines $\rightarrow$ [`docs/architecture.md`](architecture.md)
- Empirical evaluations $\rightarrow$ [`docs/evaluation.md`](evaluation.md)
- System constraints & failure modes $\rightarrow$ [`docs/limitations.md`](limitations.md)
- Public function signatures $\rightarrow$ [`docs/api.md`](api.md)

---

## 4. Code Quality & Typing Standards

Contributors are expected to adhere to standard Python PEP 8 conventions:

- **Type Hints**: All function arguments and return types must include explicit type annotations:
  ```python
  from typing import Dict, List, Optional, Tuple, Union
  from pathlib import Path
  import torch
  ```
- **Expressive Naming**: Use descriptive, self-documenting variable and function names. Avoid cryptic single-letter abbreviations (e.g., use `class_weights_tensor` instead of `w`, or `image_tensor` instead of `x`).
- **Function Cohesion**: Keep functions focused on a single operational task. If a function exceeds ~50 lines of execution logic, decompose it into smaller, testable private helper utilities (e.g., `_helper_name()`).
- **Unicode & Cross-Platform Support**: Always ensure terminal outputs support Unicode/Arabic strings safely using:
  ```python
  if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
      try:
          sys.stdout.reconfigure(encoding="utf-8")
      except Exception:
          pass
  ```

---

## 5. Pre-Commit Verification & Sanity Checks

Before staging or committing any code changes:

1. **Mathematical Validation**: If modifying equations (e.g., Shannon entropy calculation, class weighting formula, evaluation metric aggregation), manually calculate expected values for edge cases (e.g., $p=[1.0, 0.0]$, $p=[0.5, 0.5]$, zero-division guards) and verify numerical equivalence.
2. **Direct Execution Check**: Run the affected script directly from the terminal to ensure zero syntax errors, type mismatches, or missing imports:
   ```bash
   python src/model/model.py
   python src/evaluation/trust_report.py
   ```
3. **Reproducibility Guarantee**: Ensure random seeds (`random_state=42`, `torch.manual_seed(42)`) are respected in data-splitting and weight initialization routines.

---

## 6. Commit Message Conventions

Insight follows structured, informative commit standards to maintain a clean and traceable project history:

### Format:
```text
type: concise imperative summary of the change

- Bullet point 1 explaining the motivation or engineering context
- Bullet point 2 detailing specific files or parameters modified
- Bullet point 3 referencing verification tests or mathematical checks
```

### Allowed Types:
- `feat`: A new feature or processing capability (e.g., adding a new metric).
- `fix`: A bug fix or correction (e.g., resolving a key name mismatch).
- `docs`: Documentation updates or additions in `docs/` or `README.md`.
- `refactor`: Code restructurings that neither add features nor fix bugs.
- `chore`: Repository maintenance, `.gitignore`, dependency bumps, or licensing.

### Example:
```bash
git commit -m "docs: add local development and contributing guides" \
           -m "- Document prerequisites, installation steps, and dataset structuring in development.md" \
           -m "- Specify engineering standards, docstring rules, and commit formats in contributing.md" \
           -m "- Reiterate ethical clinician-in-the-loop boundaries for all future medical additions"
```

---

## 7. Ethical & Clinical Safety Constraints

Because Insight operates within the pediatric healthcare domain, all contributors must observe strict ethical boundaries:

> [!CAUTION]
> **Mandatory Review Against Clinical Guidelines**
>
> Any pull request or code change that introduces, modifies, or formats patient-facing or physician-facing text (such as clinical decision recommendations, uncertainty alerts, or diagnostic labels) **must be reviewed against [`docs/clinical-guidelines.md`](clinical-guidelines.md) prior to merging.**

### Mandatory Clinical Constraints:
1. **Never Claim Autonomous Diagnostic Authority**: Insight is strictly a **Clinical Decision Support System (CDSS)**. Any output phrasing suggesting the model makes independent medical diagnoses is strictly prohibited.
2. **Preserve Clinician-in-the-Loop Safeguards**: Outputs must emphasize that decisions require human clinical correlation (symptoms, auscultation, lab panels).
3. **Escalation Phrasing Alignment**: Ensure clinical escalation text strictly matches approved medical terminology (e.g., *"مطلوب مراجعة إلزامية من طبيب أشعة استشاري قبل اعتماد النتيجة"* / *"Mandatory senior radiologist review required prior to sign-off"*).

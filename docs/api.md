<div align="center">

# API Reference & Technical Specification
### Insight: Pediatric Pneumonia Detection & Clinical Decision Support System

[![Language](https://img.shields.io/badge/Language-Python%203.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Type](https://img.shields.io/badge/Type-API%20Reference-brightgreen?style=flat-square)](#table-of-contents)

[Data Preparation](#1-srcdataprepare_datasetpy) • [Dataset & Loaders](#2-srcdatadatasetpy) • [Model Architecture](#3-srcmodelmodelpy) • [Training Pipeline](#4-srcmodeltrainpy) • [Explainability](#5-srcexplainabilitygradcampy) • [Uncertainty](#6-srcevaluationuncertaintypy) • [Trust Report](#7-srcevaluationtrust_reportpy) • [Model Evaluation](#8-srcevaluationevaluate_modelpy)

</div>

---

## Overview

This document provides a comprehensive technical API reference for all public classes, functions, and utilities implemented across the Insight codebase. Modules are organized according to the system's end-to-end processing pipeline:

```text
Data Preparation ──► Dataset / DataLoaders ──► Model Definition ──► Training Pipeline
                                                                           │
Benchmark Evaluation ◄── Trust Report Orchestrator ◄── Uncertainty + Grad-CAM
```

---

## 1. `src/data/prepare_dataset.py`

Handles raw dataset directory scanning, resolves Kaggle's 16-image validation set limitation, executes stratified 70/15/15 development pool partitioning, and preserves the official isolated test benchmark.

### `scan_dataset_directory`
```python
def scan_dataset_directory(raw_dir: Path) -> List[Dict[str, Any]]
```
Recursively traverses the raw Kaggle dataset root directory (`train/`, `val/`, `test/` subfolders) and collects metadata records for all valid radiographs.

- **Parameters**:
  - `raw_dir` (`pathlib.Path`): Path to the raw dataset root directory (e.g., `data/raw/chest_xray`).
- **Returns**:
  - `List[Dict[str, Any]]`: List of dictionaries containing `'filepath'` (relative POSIX path), `'label'` (0 for NORMAL, 1 for PNEUMONIA), and `'source_split'` (`'train'`, `'val'`, or `'test'`).

### `perform_stratified_split`
```python
def perform_stratified_split(
    records: List[Dict[str, Any]], 
    random_state: int = 42
) -> pd.DataFrame
```
Aggregates Kaggle's `train` and `val` folders into a 5,232-image development pool, applies a stratified 70/15/15 split (`train`, `val`, `internal_test`), and preserves the 624-image `test` directory as `official_test`.

- **Parameters**:
  - `records` (`List[Dict[str, Any]]`): Image metadata dictionaries produced by `scan_dataset_directory`.
  - `random_state` (`int`, default=`42`): Seed for deterministic, reproducible pseudorandom splitting.
- **Returns**:
  - `pd.DataFrame`: Formatted manifest DataFrame with columns `['filepath', 'label', 'split']`.

### `print_statistical_summary`
```python
def print_statistical_summary(df: pd.DataFrame) -> None
```
Outputs a formatted terminal table summarizing sample volumes, class distributions, percentages, and class imbalance ratios across all splits.

- **Parameters**:
  - `df` (`pd.DataFrame`): The processed manifest DataFrame.

```python
# Example Usage:
from pathlib import Path
from src.data.prepare_dataset import scan_dataset_directory, perform_stratified_split

records = scan_dataset_directory(Path("data/raw/chest_xray"))
manifest_df = perform_stratified_split(records, random_state=42)
manifest_df.to_csv("data/processed/manifest.csv", index=False)
```

---

## 2. `src/data/dataset.py`

PyTorch `Dataset` and `DataLoader` pipeline handling resilient on-the-fly data loading, augmentations, and tensor normalization.

### `ChestXrayDataset`
```python
class ChestXrayDataset(torch.utils.data.Dataset)
```
Custom PyTorch Dataset for loading pediatric chest radiographs from `manifest.csv`. Features automatic error-recovery against corrupted images with fallback sampling.

#### `__init__`
```python
def __init__(
    self,
    manifest_path: Union[str, Path],
    split: str,
    transform: Optional[Callable] = None,
    root_dir: Optional[Union[str, Path]] = None,
) -> None
```
- **Parameters**:
  - `manifest_path` (`str` | `Path`): Path to `data/processed/manifest.csv`.
  - `split` (`str`): Target split to load (`'train'`, `'val'`, `'internal_test'`, or `'official_test'`).
  - `transform` (`Optional[Callable]`, default=`None`): Torchvision transformation pipeline applied to PIL images.
  - `root_dir` (`Optional[str | Path]`, default=`None`): Base directory for resolving relative file paths.

#### `__len__`
```python
def __len__(self) -> int
```
Returns the total count of valid samples available in the filtered split.

#### `__getitem__`
```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]
```
Loads the radiograph at row `idx`, forces RGB 3-channel conversion, applies transforms, and returns the tensor paired with its integer ground-truth label.

- **Returns**:
  - `Tuple[torch.Tensor, int]`: Tuple of `(transformed_image_tensor, integer_label)`.

### `get_transforms`
```python
def get_transforms(split: str = "train") -> torchvision.transforms.Compose
```
Builds torchvision transform pipelines aligned with medical imaging constraints (no vertical flips):
- **Train**: Resize to $(224, 224)$, random affine rotation ($\pm 10^\circ$), horizontal flip ($p=0.5$), tensor conversion, ImageNet standardization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
- **Val / Test**: Resize to $(224, 224)$, tensor conversion, ImageNet standardization.

- **Parameters**:
  - `split` (`str`, default=`"train"`): Split identifier (`"train"`, `"val"`, or `"test"`).
- **Returns**:
  - `transforms.Compose`: Configured torchvision transformation pipeline.

### `get_dataloaders`
```python
def get_dataloaders(
    manifest_path: Union[str, Path] = "data/processed/manifest.csv",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, torch.utils.data.DataLoader]
```
Convenience factory creating PyTorch DataLoaders across all four splits simultaneously.

- **Parameters**:
  - `manifest_path` (`str` | `Path`, default=`"data/processed/manifest.csv"`): Path to manifest CSV.
  - `batch_size` (`int`, default=`32`): Mini-batch size.
  - `num_workers` (`int`, default=`0`): Subprocess worker count (0 recommended for Windows).
  - `pin_memory` (`bool`, default=`False`): Enables CUDA page-locked memory allocation.
- **Returns**:
  - `Dict[str, DataLoader]`: Dictionary keyed by split name (`'train'`, `'val'`, `'internal_test'`, `'official_test'`).

```python
# Example Usage:
from src.data.dataset import get_dataloaders

loaders = get_dataloaders(batch_size=32)
for images, labels in loaders["train"]:
    print(images.shape)  # torch.Size([32, 3, 224, 224])
    break
```

---

## 3. `src/model/model.py`

Constructs the DenseNet-121 architecture, executes two-stage parameter freezing, and configures the binary classification head.

### `build_densenet121`
```python
def build_densenet121(pretrained: bool = True) -> nn.Module
```
Instantiates a DenseNet-121 backbone using torchvision weights (`DenseNet121_Weights.IMAGENET1K_V1`) and substitutes the default 1000-class classifier with a 2-class binary head:
```python
nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features=1024, out_features=2)
)
```

- **Parameters**:
  - `pretrained` (`bool`, default=`True`): Whether to load pre-trained ImageNet weights.
- **Returns**:
  - `nn.Module`: Modified DenseNet-121 PyTorch network.

### `freeze_early_layers`
```python
def freeze_early_layers(model: nn.Module) -> nn.Module
```
Enforces the two-stage transfer learning policy:
- **Frozen (`requires_grad = False`)**: `conv0`, `norm0`, `denseblock1`, `transition1`, `denseblock2`, `transition2`, `denseblock3`, `transition3`.
- **Trainable (`requires_grad = True`)**: `denseblock4`, `norm5`, `classifier`.

- **Parameters**:
  - `model` (`nn.Module`): DenseNet-121 model instance.
- **Returns**:
  - `nn.Module`: Model with selectively frozen weights.

### `count_parameters`
```python
def count_parameters(model: nn.Module) -> Tuple[int, int, float]
```
Calculates total parameter counts and computes the percentage of trainable parameters.

- **Parameters**:
  - `model` (`nn.Module`): PyTorch model.
- **Returns**:
  - `Tuple[int, int, float]`: `(trainable_parameters, total_parameters, trainable_percentage)`.

### `build_model`
```python
def build_model(device: Optional[Union[str, torch.device]] = None) -> nn.Module
```
Primary factory pipeline: instantiates DenseNet-121, applies layer freezing, and allocates weights to the designated hardware device.

- **Parameters**:
  - `device` (`Optional[str | torch.device]`, default=`None`): Target compute device (`'cuda'` or `'cpu'`).
- **Returns**:
  - `nn.Module`: Configured model ready for training or inference.

```python
# Example Usage:
import torch
from src.model.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(device=device)
```

---

## 4. `src/model/train.py`

Training and validation loops with class-weighted loss, decoupled weight decay, learning rate scheduling, and early stopping.

### `compute_class_weights`
```python
def compute_class_weights(
    manifest_path: Union[str, Path] = "data/processed/manifest.csv",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor
```
Computes balanced class weights on the training split using the inverse-frequency formulation:
$$w_c = \frac{N}{C \cdot N_c}$$

- **Parameters**:
  - `manifest_path` (`str` | `Path`): Path to manifest CSV.
  - `device` (`Optional[str | torch.device]`): Device to transfer the tensor to.
- **Returns**:
  - `torch.Tensor`: 1D tensor of shape `(2,)` containing class weights $[w_{\text{Normal}}, w_{\text{Pneumonia}}]$.

### `train_one_epoch`
```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float
```
Performs a single complete training epoch: forward pass, loss calculation, backpropagation, and parameter optimization.

- **Parameters**:
  - `model` (`nn.Module`): DenseNet-121 model.
  - `loader` (`DataLoader`): Training DataLoader.
  - `criterion` (`nn.Module`): Loss function (`nn.CrossEntropyLoss` with weights).
  - `optimizer` (`torch.optim.Optimizer`): Optimizer (`AdamW`).
  - `device` (`torch.device`): Active compute device.
- **Returns**:
  - `float`: Average training loss across the epoch.

### `evaluate`
```python
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]
```
Evaluates loss and accuracy on a DataLoader without computing gradients (`torch.no_grad()`).

- **Parameters**:
  - `model` (`nn.Module`): PyTorch model.
  - `loader` (`DataLoader`): Evaluation DataLoader (`val` or `test`).
  - `criterion` (`nn.Module`): Loss function.
  - `device` (`torch.device`): Active compute device.
- **Returns**:
  - `Tuple[float, float]`: `(average_loss, accuracy_fraction)`.

---

## 5. `src/explainability/gradcam.py`

Generates gradient-weighted class activation heatmaps to interpret model attention spatially.

### `generate_gradcam`
```python
def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray
```
Computes a normalized 2D Grad-CAM heatmap targeting the final convolutional layer of DenseNet-121 (`model.features.denseblock4.denselayer16.conv2`).

- **Parameters**:
  - `model` (`nn.Module`): DenseNet-121 model.
  - `image_tensor` (`torch.Tensor`): Input tensor of shape `(1, 3, 224, 224)` or `(3, 224, 224)`.
  - `target_layer` (`Optional[nn.Module]`, default=`None`): Target layer for gradient extraction (defaults to DenseBlock4 Conv2).
  - `device` (`Optional[str | torch.device]`, default=`None`): Compute device.
- **Returns**:
  - `np.ndarray`: 2D float array of shape `(224, 224)` containing normalized saliency activations $[0.0, 1.0]$.

### `overlay_heatmap`
```python
def overlay_heatmap(
    original_image_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image
```
Resizes and blends the 2D Grad-CAM heatmap over the original radiograph using a JET color map.

- **Parameters**:
  - `original_image_pil` (`PIL.Image.Image`): Raw input radiograph.
  - `heatmap` (`np.ndarray`): 2D activation array $[0.0, 1.0]$.
  - `alpha` (`float`, default=`0.5`): Transparency weighting ($0.0 = \text{image only}$, $1.0 = \text{heatmap only}$).
- **Returns**:
  - `PIL.Image.Image`: RGB image containing the blended diagnostic overlay.

### `analyze_salient_region`
```python
def analyze_salient_region(heatmap: np.ndarray) -> str
```
Analyzes spatial distribution of peak activations across anatomical lung zones (Apical/Upper, Mid-zone, Basilar/Lower) and hemithorax sides (Left, Right).

- **Parameters**:
  - `heatmap` (`np.ndarray`): 2D activation array.
- **Returns**:
  - `str`: Clinical anatomical description string (e.g., `"Right lung / Right hemithorax (Mid-zone)"`).

```python
# Example Usage:
from PIL import Image
from src.data.dataset import get_transforms
from src.explainability.gradcam import generate_gradcam, overlay_heatmap

pil_img = Image.open("data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")
tensor = get_transforms("val")(pil_img).unsqueeze(0)
heatmap = generate_gradcam(model, tensor)
overlay = overlay_heatmap(pil_img, heatmap, alpha=0.5)
overlay.save("outputs/reports/sample_overlay.png")
```

---

## 6. `src/evaluation/uncertainty.py`

Predictive uncertainty quantification module based on Shannon entropy with automated mapping to operational clinical tiers.

### `compute_entropy`
```python
def compute_entropy(
    probabilities: Union[torch.Tensor, np.ndarray, List[float]],
    eps: float = 1e-12,
) -> float
```
Computes Normalized Shannon Entropy for binary class probabilities:
$$H(p) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
Naturally bounded in $[0.0, 1.0]$ for $C=2$. Handles zero-probability edge cases safely via masking.

- **Parameters**:
  - `probabilities` (`Tensor` | `ndarray` | `List[float]`): Probability distribution summing to $1.0$.
  - `eps` (`float`, default=`1e-12`): Epsilon to prevent $\log_2(0)$.
- **Returns**:
  - `float`: Normalized entropy value in range $[0.0, 1.0]$.

### `get_uncertainty_tier`
```python
def get_uncertainty_tier(entropy_value: float) -> str
```
Maps normalized Shannon entropy into operational triage tiers:
- **`'Low'`**: $\hat{H} < 0.30$ (Clear, prototypical prediction).
- **`'Medium'`**: $0.30 \le \hat{H} < 0.70$ (Borderline density; dual review recommended).
- **`'High'`**: $\hat{H} \ge 0.70$ (Marginal confidence; mandatory escalation).

- **Parameters**:
  - `entropy_value` (`float`): Normalized entropy value.
- **Returns**:
  - `str`: Tier label (`'Low'`, `'Medium'`, or `'High'`).

### `predict_with_uncertainty`
```python
def predict_with_uncertainty(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Union[int, float, str, List[float]]]
```
Performs forward inference, extracts softmax probabilities, computes Shannon entropy, and determines the uncertainty tier.

- **Parameters**:
  - `model` (`nn.Module`): DenseNet-121 model.
  - `image_tensor` (`torch.Tensor`): Input tensor of shape `(1, 3, 224, 224)`.
  - `device` (`Optional[str | torch.device]`): Compute device.
- **Returns**:
  - `Dict[str, Any]` with keys:
    - `'predicted_class'` (`int`): $0$ (NORMAL) or $1$ (PNEUMONIA).
    - `'class_name'` (`str`): `'NORMAL'` or `'PNEUMONIA'`.
    - `'probability'` (`float`): Confidence in predicted class.
    - `'probabilities'` (`List[float]`): $[p_{\text{Normal}}, p_{\text{Pneumonia}}]$.
    - `'entropy'` (`float`): Normalized Shannon entropy.
    - `'tier'` (`str`): `'Low'`, `'Medium'`, or `'High'`.

```python
# Example Usage:
from src.evaluation.uncertainty import predict_with_uncertainty

result = predict_with_uncertainty(model, tensor)
print(f"Prediction: {result['class_name']} ({result['probability']*100:.2f}%)")
print(f"Entropy: {result['entropy']:.4f} | Tier: {result['tier']}")
```

---

## 7. `src/evaluation/trust_report.py`

Central orchestration engine synthesizing classification probabilities, Grad-CAM saliency overlays, predictive uncertainty, and actionable clinical recommendations into unified multimodal artifacts.

### `format_report_text`
```python
def format_report_text(report_data: Dict[str, Any]) -> str
```
Formats structured trust report dictionaries into a readable ASCII clinical document.

- **Parameters**:
  - `report_data` (`Dict[str, Any]`): Report dictionary from `generate_trust_report`.
- **Returns**:
  - `str`: Multi-line formatted clinical report text.

### `generate_trust_report`
```python
def generate_trust_report(
    model: nn.Module,
    image_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    save_dir: Union[str, Path] = "outputs/reports",
) -> Dict[str, Any]
```
Executes complete clinical inference and generates multimodal artifacts:
1. Preprocesses image through validation transform pipeline.
2. Infers prediction probabilities, entropy, and uncertainty tier.
3. Computes Grad-CAM attention heatmap and generates RGB overlay image.
4. Formulates clinical decision guidance mapped to the uncertainty tier.
5. Saves 3 artifacts:
   - `<image_stem>_gradcam.png` (Visual Saliency Overlay)
   - `<image_stem>_report.json` (Structured EHR Interoperability Schema)
   - `<image_stem>_report.txt` (Human-readable Clinical Text Document)

- **Parameters**:
  - `model` (`nn.Module`): Trained DenseNet-121 model.
  - `image_path` (`str` | `Path`): Path to target radiograph file.
  - `device` (`Optional[str | torch.device]`): Compute device.
  - `save_dir` (`str` | `Path`, default=`"outputs/reports"`): Destination directory.
- **Returns**:
  - `Dict[str, Any]`: Dictionary containing all diagnostic values, paths, and clinical directives.

```python
# Example Usage:
from src.evaluation.trust_report import generate_trust_report

report = generate_trust_report(
    model=model,
    image_path="data/raw/chest_xray/train/PNEUMONIA/person635_bacteria_2526.jpeg",
    save_dir="outputs/reports"
)
print("Saved report to:", report["text_report_path"])
```

---

## 8. `src/evaluation/evaluate_model.py`

Comprehensive offline evaluation engine assessing global benchmark performance on the held-out test set (`official_test`, $N=624$).

### `full_evaluation`
```python
def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]
```
Runs batch inference over a complete DataLoader and computes all clinical diagnostic metrics:
- Overall Accuracy ($\frac{TP+TN}{N}$)
- Sensitivity / Recall ($\frac{TP}{TP+FN}$)
- Specificity ($\frac{TN}{TN+FP}$)
- Precision / PPV ($\frac{TP}{TP+FP}$)
- F1-Score ($2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec}+\text{Rec}}$)
- AUC-ROC (Area under the Receiver Operating Characteristic curve)
- Complete $2 \times 2$ Confusion Matrix ($TN, FP, FN, TP$)

- **Parameters**:
  - `model` (`nn.Module`): Trained model checkpoint.
  - `loader` (`DataLoader`): DataLoader for evaluation split (`official_test`).
  - `device` (`Optional[str | torch.device]`): Compute device.
- **Returns**:
  - `Dict[str, Any]`: Dictionary containing scalar metrics, raw target arrays, prediction arrays, and confusion matrix counts.

### `plot_and_save_confusion_matrix`
```python
def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    save_path: Union[str, Path] = "outputs/reports/confusion_matrix.png",
    class_names: Optional[List[str]] = None,
) -> Path
```
Renders a publication-quality confusion matrix visualization using Matplotlib and saves the PNG artifact to disk.

- **Parameters**:
  - `cm` (`np.ndarray`): $2 \times 2$ confusion matrix integer array.
  - `save_path` (`str` | `Path`, default=`"outputs/reports/confusion_matrix.png"`): File path for plot.
  - `class_names` (`Optional[List[str]]`, default=`['NORMAL', 'PNEUMONIA']`): Class labels.
- **Returns**:
  - `pathlib.Path`: Path to the saved image file.

```python
# Example Usage:
from src.data.dataset import get_dataloaders
from src.evaluation.evaluate_model import full_evaluation, plot_and_save_confusion_matrix

loaders = get_dataloaders(batch_size=32)
metrics = full_evaluation(model, loaders["official_test"])
print(f"Accuracy: {metrics['accuracy']*100:.2f}% | Sensitivity: {metrics['recall']*100:.2f}%")
plot_and_save_confusion_matrix(metrics["confusion_matrix"])
```

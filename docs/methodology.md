# Methodology & Technical Implementation

## 1. Model Architecture

### Backbone: DenseNet-121 Pretrained on ImageNet
Insight adopts **DenseNet-121** (Huang et al., 2017) initialized with ImageNet pre-trained weights as the core feature extractor.

```text
Input Image (224x224x3)
        │
   [conv0 + pool0]           (Frozen)
        │
   [DenseBlock 1] + [Trans1]  (Frozen)
        │
   [DenseBlock 2] + [Trans2]  (Frozen)
        │
   [DenseBlock 3] + [Trans3]  (Frozen)
        │
   [DenseBlock 4]            (Trainable / Fine-tuned)
        │
      [norm5]                (Trainable)
        │
   Global Average Pooling
        │
   Classifier Head           (New Linear Head: 1024 -> 2 classes)
```

### Literature Justification (CheXNet Benchmark)
The choice of DenseNet-121 is grounded directly in medical imaging literature:
- **CheXNet (Rajpurkar et al., Stanford, 2017)** demonstrated that a 121-layer DenseNet outperforms average practicing radiologists in detecting pneumonia from frontal chest radiographs.
- **Dense Connectivity**: Each layer receives direct inputs from all preceding layers. This architectural feature reuse preserves both low-level structural details (rib contours, diaphragm edges) and high-level pathological patterns (diffuse infiltrates, consolidation).
- **Gradient Flow & Efficiency**: Dense connections alleviate vanishing gradient problems and maintain significantly fewer parameters (~7M) compared to ResNet-50 (~25M) or VGG-16 (~138M), mitigating overfitting on moderately sized medical datasets.

### Transfer Learning & Fine-Tuning Strategy
A two-stage parameter unfreezing strategy ensures feature stability while adapting to thoracic radiographs:
1. **Early Feature Freezing**: The initial convolutional layer (`features.conv0`) and the first three dense blocks (`denseblock1` through `denseblock3` along with their transition layers) are frozen. These layers capture general visual primitives (edges, textures, basic geometry) learned from ImageNet.
2. **Deep Adaptation**: The final dense block (`features.denseblock4`), the final batch normalization layer (`features.norm5`), and the newly attached classification head are kept trainable to specialize in domain-specific pulmonary opacities.
3. **Custom Classifier Head**: The original 1000-class ImageNet linear layer is replaced with a custom binary classification block:
   ```python
   nn.Sequential(
       nn.Dropout(p=0.3),
       nn.Linear(in_features=1024, out_features=2)  # [Normal, Pneumonia]
   )
   ```

---

## 2. Dataset Splitting Strategy

### The Kaggle Default Split Issue
The default Kaggle Chest X-ray (Pneumonia) dataset distribution contains a critical structural flaw:
- `train/`: 5,216 images
- `test/`: 624 images
- `val/`: **16 images only** (8 Normal, 8 Pneumonia)

A validation set of 16 images introduces extreme sampling variance: a single misclassified scan alters validation accuracy by $6.25\%$. This makes validation loss completely unrepresentative and renders early stopping mechanisms unusable.

### Re-Splitting Protocol
To establish a statistically robust evaluation framework:

```text
Original Kaggle Distribution:
┌───────────────────────────────────────┐ ┌────────┐ ┌────────┐
│             train (5,216)             │ │val (16)│ │test (624)
└───────────────────────────────────────┘ └────────┘ └────────┘
                   │                          │           │
                   └───────────┬──────────────┘           │
                               ▼                          ▼
                     Development Pool (5,232)      Held-Out Test
                               │                      (624)
            ┌──────────────────┼──────────────────┐   [Untouched]
            ▼                  ▼                  ▼
       Train (70%)         Val (15%)          Test (15%)
      (~3,662 scans)      (~785 scans)       (~785 scans)
```

1. **Development Pool Aggregation**: The original `train` (5,216) and `val` (16) sets are merged into a single pool of 5,232 images.
2. **Stratified Partitioning**: The combined pool is partitioned into:
   - **Training Set (70%)**: Used for gradient updates.
   - **Validation Set (15%)**: Used for hyperparameter tuning, metric tracking, and early stopping.
   - **Internal Test Set (15%)**: Used for unbiased internal model checkpoint selection.
3. **Preserved Benchmark Test Set**: The official Kaggle `test` set (624 images) remains strictly isolated as a **held-out external benchmark**. It is evaluated only once after final model selection to verify generalizability.

---

## 3. Handling Class Imbalance

### Class Distribution
The training set exhibits an inherent medical skew:
- **Pneumonia**: ~74% (~3,875 scans)
- **Normal**: ~26% (~1,357 scans)
- Ratio: approximately **3:1** (Pneumonia to Normal)

### Weighted Cross-Entropy Loss vs. Oversampling
We address this disparity using **Class-Weighted Cross-Entropy Loss** rather than data oversampling:

$$\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} w_{y_i} \log(p_{i, y_i})$$

Where class weights $w_c$ are calculated inversely proportional to class frequencies:
$$w_c = \frac{N}{C \cdot N_c}$$
- $N$: Total number of training samples.
- $C$: Number of classes ($C=2$).
- $N_c$: Number of samples in class $c$.

### Why Weighted Loss Over Oversampling?
1. **Prevents Overfitting on Normal Scans**: Oversampling (duplicating minority Normal images) leads convolutional neural networks to memorize specific patient anatomies and radiographic artifacts, degrading generalization.
2. **Preserves Data Manifold**: Oversampling artificially repeats high-gradient instances in mini-batches. Class weighting mathematically scales the gradient of underrepresented samples without altering the true underlying data manifold.
3. **Computational Efficiency**: Avoids expanding epoch duration and disk/memory footprint.

---

## 4. Training Configuration & Hyperparameters

### Data Augmentation Pipeline
To reflect natural variation in radiographic acquisition while preserving anatomical validity:
- **Input Resize**: Resized to $224 \times 224$ pixels.
- **Random Affine / Rotation**: $\pm 10^\circ$ (simulating minor patient tilt in pediatric imaging).
- **Horizontal Flip**: $p = 0.5$ (thoracic bilateral symmetry).
- **Strictly Prohibited**: **No vertical flipping** (thoracic anatomy maintains strict apical-to-basilar cranial-caudal orientation).
- **Normalization**: Standardized using ImageNet statistics:
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$

### Optimization & Hyperparameters

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Optimizer** | AdamW | Decoupled weight decay provides superior regularization over standard Adam. |
| **Learning Rate** | $1 \times 10^{-4}$ | Low learning rate prevents destabilizing pretrained backbone weights. |
| **Weight Decay** | $1 \times 10^{-2}$ | Regularizes fine-tuned parameters in denseblock4 and head. |
| **Batch Size** | 32 | Balances gradient estimation stability with GPU memory constraints. |
| **Epochs** | 20 (Max) | Convergence typically achieved within 12–16 epochs. |
| **LR Scheduler** | `ReduceLROnPlateau` | Factor 0.5, patience 2 epochs based on validation loss. |
| **Early Stopping** | Patience = 5 epochs | Monitors validation loss to prevent overfitting. |

---

## 5. Explainability: Grad-CAM Implementation

### Framework & Objective
Insight integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** using the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library to generate visual explanations for each prediction.

### Mathematical Formulation
Grad-CAM calculates the importance of each feature activation map $A^k$ for target class $c$:

1. **Neuron Importance Weights ($\alpha_k^c$)**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i, j}^k}$$
   Where $Y^c$ is the pre-softmax score for class $c$, $A_{i, j}^k$ is the activation at spatial position $(i, j)$ of feature map $k$, and $Z$ is the spatial area of the feature map.

2. **Heatmap Generation ($L_{\text{Grad-CAM}}^c$)**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   The $\text{ReLU}$ operator retains only features that have a positive influence on the target class score, filtering out features that contribute to alternate classes.

### Target Layer in DenseNet-121
The designated target layer is:
```text
model.features.denseblock4.denselayer16.conv2
```
(Alternatively accessed via `model.features.norm5` prior to global average pooling).
- **Why this layer?** This represents the final convolutional stage of the architecture. It captures the richest semantic abstractions (lobar consolidations, interstitial infiltrates) while still preserving spatial coordinate mapping ($7 \times 7$ grid at $224 \times 224$ input resolution) before pooling collapses spatial dimensionality.

---

## 6. Predictive Uncertainty Estimation

### Shannon Entropy Formulation
To ensure safe clinical integration, Insight computes an **entropy-based uncertainty score** from the post-softmax predictive probabilities $p = [p_{\text{Normal}}, p_{\text{Pneumonia}}]$.

Standard Shannon Entropy for $C=2$:
$$H(p) = - \sum_{i=1}^{C} p_i \log_2(p_i) = - \left( p_0 \log_2(p_0) + p_1 \log_2(p_1) \right)$$

Because $\max(H) = \log_2(2) = 1.0$ for binary classification, the raw entropy is naturally normalized on the range $[0, 1]$:
$$\hat{H}(p) = H(p) \in [0, 1]$$

- When the model is completely confident ($p = [1.0, 0.0]$ or $[0.0, 1.0]$): $\hat{H} = 0.0$.
- When the model is completely uncertain ($p = [0.5, 0.5]$): $\hat{H} = 1.0$.

### Uncertainty Tier Categorization & Clinical Protocol
In alignment with the clinical protocol defined in [`docs/clinical-guidelines.md`](clinical-guidelines.md#5-bridging-model-uncertainty-to-clinical-practice), predictions are mapped into three actionable operational tiers:

```text
        0.0                         0.3                         0.7                         1.0
Entropy: ├───────────────────────────┼───────────────────────────┼───────────────────────────┤
Tier:    │      Low Uncertainty      │    Medium Uncertainty     │     High Uncertainty      │
Action:  │   Standard Triage Flow    │   Flag for Dual Check     │ Mandatory Second-Reader   │
```

| Tier | Normalized Entropy ($\hat{H}$) | Primary Class Probability | Clinical Operational Action |
| :--- | :--- | :--- | :--- |
| **Low Uncertainty** | $\hat{H} < 0.30$ | $> 94\%$ / $< 6\%$ | Standard clinical queue. Grad-CAM visual check by primary reader. |
| **Medium Uncertainty** | $0.30 \le \hat{H} < 0.70$ | $80\% - 94\%$ / $6\% - 20\%$ | Secondary review recommended. Correlate with physical exam and symptoms. |
| **High Uncertainty** | $\hat{H} \ge 0.70$ | $< 80\%$ (Marginal) | **Mandatory escalation**: Flagged for senior radiologist review, clinical lab correlation (CBC, CRP), or repeat radiograph if technical artifacts exist. |

This quantitative gate prevents silent algorithmic failures and guarantees that ambiguous or out-of-distribution cases are escalated to expert clinical oversight.

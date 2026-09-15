# Insight: Medical Image Prediction with Explainability & Uncertainty

An end-to-end deep learning system designed for medical image classification (Chest X-ray Pneumonia detection) integrating interpretability (Grad-CAM) and predictive uncertainty estimation to serve as a reliable clinical decision support tool.

## Project Structure

```text
Insight/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── model/
│   ├── explainability/
│   └── evaluation/
├── notebooks/
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── evaluation.md
│   ├── limitations.md
│   ├── api.md
│   ├── development.md
│   ├── contributing.md
│   └── clinical-guidelines.md
├── tests/
├── outputs/
│   ├── gradcam/
│   └── reports/
├── requirements.txt
├── .gitignore
└── README.md
```

## Documentation

Detailed technical and clinical documentation is available in the [`docs/`](docs/) directory:
- [Clinical Guidelines & Scope](docs/clinical-guidelines.md)
- [System Architecture](docs/architecture.md)
- [Methodology & Modeling](docs/methodology.md)
- [Evaluation Framework](docs/evaluation.md)
- [Limitations & Failure Modes](docs/limitations.md)
- [API & Inference Reference](docs/api.md)
- [Development Setup](docs/development.md)
- [Contributing Guidelines](docs/contributing.md)

## Disclaimer

This project is intended strictly for educational, research, and portfolio purposes. It is **not** a certified medical diagnostic device and must not be used for actual clinical diagnosis.

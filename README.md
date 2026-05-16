# 🛰️ Deep Learning-Based Cloud Pattern Segmentation from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)

Semantic segmentation of four mesoscale cloud organization patterns — **Sugar, Flower, Fish, and Gravel** — from NASA satellite imagery. Built on a **U-Net + EfficientNet-B1** architecture, this project benchmarks five state-of-the-art segmentation models and achieves a **Kaggle private leaderboard score of 0.66479**, approaching the competition winner (0.67175) with a single model while the winner relied on a 9-model ensemble.

> Published as a research paper: *"Deep Learning-Based Cloud Pattern Segmentation from Satellite Imagery"* — Utku Efe Akdoğan & Fatih Alagöz, Boğaziçi University.  
> Based on the [Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization) Kaggle competition.

---

## 📊 Results

| | Score |
|--|-------|
| 🥇 Competition Winner (9-model ensemble) | 0.67175 |
| **This model (single U-Net + EfficientNet-B1)** | **0.66479** |

> Mean Dice: **0.6543** — IoU: **0.6067**. The competition winner used an ensemble of 9 segmentation models with 3 classification heads. Our single-model approach closes 97% of the gap while remaining deployable on standard GPU infrastructure.

---

## 🏗️ Architecture

The U-Net encoder was replaced with **EfficientNet-B1 (7.8M parameters)** for improved feature extraction efficiency. Skip connections between encoder and decoder preserve fine-grained spatial information — critical for detecting subtle cloud boundaries.

### Architectures Benchmarked

| Model | Training Time | Notes |
|-------|---------------|-------|
| **U-Net + EfficientNet-B1** | 6–8 hrs | Best performance, optimal balance |
| FPN | 8–10 hrs | Close second, higher cost |
| MAnet | 10–12 hrs | Attention overhead without gain |
| UNet++ | 10–12 hrs | Dense skip paths didn't help with cloud ambiguity |
| DeepLab V3+ | 10–12 hrs | Rapid overfitting, unstable training |

**Key finding:** Architectural complexity does not improve performance in domains with inherently ambiguous ground truth. The boundary between cloud pattern types is subjective even to expert meteorologists — a fundamental ceiling no architecture can overcome alone.

---

## 🔬 Methodology

- **Loss:** Binary Cross-Entropy + Dice Loss
- **Augmentation:** Horizontal/vertical flips, geometric distortions, noise injection
- **Threshold Optimization:** Random sampling over 100 combinations (label: 0.80–0.95, pixel: 0.20–0.45 per class)
- **Training:** Dual Tesla T4 GPUs, batch size 16, mixed-precision via PyTorch `DataParallel`

---

## 🚀 Getting Started

```bash
pip install torch torchvision segmentation-models-pytorch albumentations kaggle

# Download dataset (requires Kaggle API credentials)
kaggle competitions download -c understanding_cloud_organization

# Run
jupyter notebook u2net.ipynb
```

---

## 📄 Citation
Akdoğan, U.E., Alagöz, F. (2024). Deep Learning-Based Cloud Pattern Segmentation
from Satellite Imagery. Boğaziçi University.

---

## 📝 License

Licensed under the Apache License 2.0. See [LICENSE](./LICENSE) for details.

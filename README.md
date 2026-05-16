# 🛰️ Deep Learning-Based Cloud Pattern Segmentation from Satellite Imagery

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)

Semantic segmentation of four mesoscale cloud organization patterns — **Sugar, Flower, Fish, and Gravel** — from NASA satellite imagery. Built on a **U-Net + EfficientNet-B1** architecture, this project benchmarks five state-of-the-art segmentation models and achieves a **Kaggle private leaderboard score of 0.66479**, approaching the competition winner (0.67175) with a single model while the winner relied on a 9-model ensemble.

> Published as a research paper: *"Deep Learning-Based Cloud Pattern Segmentation from Satellite Imagery"* — Utku Efe Akdoğan & Fatih Alagöz, Boğaziçi University.  
> Based on the [Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization) Kaggle competition.

---

## 🌩️ Cloud Pattern Classes

The four target patterns were first identified by a team of 67 scientists who labeled ~50,000 mesoscale cloud clusters across 10,000 satellite images via the Zooniverse platform.

| Pattern | Characteristics | Segmentation Difficulty |
|---------|----------------|------------------------|
| **Flower** | Large circular structures, 50–200 km diameter, clear cloud-free gaps | ✅ Easiest — distinct boundaries |
| **Fish** | Skeletal "fishbone" structures spanning up to 1,000 km | 🔶 Intermediate |
| **Gravel** | Granular arcs/rings ~20 km in diameter | 🔶 Intermediate |
| **Sugar** | Fine-grained small cumulus, no clear boundaries | ❌ Hardest — subtle textures |

---

## 📊 Results

### Performance by Cloud Type

| Pattern | Dice Score | IoU |
|---------|-----------|-----|
| Flower  | **0.7663** | **0.7310** |
| Fish    | 0.6437 | — |
| Gravel  | 0.6125 | — |
| Sugar   | 0.5948 | 0.5235 |
| **Overall** | **0.6543** | **0.6067** |

### Kaggle Leaderboard

| | Score |
|--|-------|
| 🥇 Competition Winner (9-model ensemble) | 0.67175 |
| **This model (single U-Net + EfficientNet-B1)** | **0.66479** |
| Public leaderboard | 0.66009 |

> The competition winner used an ensemble of 9 segmentation models with 3 classification heads. Our single-model approach closes 97% of the gap while remaining deployable on standard GPU infrastructure.

---

## 🏗️ Architecture

### Chosen Model: U-Net + EfficientNet-B1

The U-Net encoder was replaced with **EfficientNet-B1 (7.8M parameters)** for improved feature extraction efficiency. Skip connections between encoder and decoder preserve fine-grained spatial information — critical for detecting subtle cloud boundaries.

```
Input (350×525 px)
    │
    ▼
EfficientNet-B1 Encoder ──── skip connections ────┐
    │                                              │
    ▼                                              │
Bottleneck                                         │
    │                                              │
    ▼                                              │
U-Net Decoder ◄────────────────────────────────────┘
    │
    ▼
Segmentation Masks (Sugar | Flower | Fish | Gravel)
```

### Architectures Benchmarked

| Model | Dice Score | Training Time | Notes |
|-------|-----------|---------------|-------|
| **U-Net + EfficientNet-B1** | **Best** | 6–8 hrs | Optimal balance |
| FPN | Close 2nd | 8–10 hrs | Good multi-scale, higher cost |
| MAnet | Below U-Net | 10–12 hrs | Attention overhead without gain |
| UNet++ | Below U-Net | 10–12 hrs | Dense skip paths didn't help with cloud ambiguity |
| DeepLab V3+ | Lowest | 10–12 hrs | Rapid overfitting, unstable training |

**Key finding:** Architectural complexity does not improve performance in domains with inherently ambiguous ground truth labels. The boundary between cloud pattern types is subjective even to expert meteorologists — a fundamental ceiling that no architecture can overcome alone.

---

## 🔬 Methodology

### Loss Function
Combined **Binary Cross-Entropy + Dice Loss**, outperforming Focal + Dice Loss for this dataset. A learnable weighted variant was tested but converged to the same values as the manually tuned fixed weights.

### Threshold Optimization
Rather than a classification head, a random sampling strategy over 100 combinations optimized label thresholds (0.80–0.95) and pixel thresholds (0.20–0.45) per class. Runtime: ~1.67 hours. Consistent high-quality solutions without exhaustive grid search.

### Data Augmentation
- Horizontal & vertical flips
- Random geometric distortions (simulate viewing angle variance)
- Noise injection (robustness)
- Resolution downsampling (computational efficiency, final output: 350×525 px)

---

## 🖥️ Training Infrastructure

| Config | GPUs | CUDA Cores | Memory | Batch Size |
|--------|------|-----------|--------|------------|
| Primary | 2× Tesla T4 | 2×2,560 | 32 GB GDDR6 | 16 |
| Secondary | Tesla P100 | 3,584 | — | 8 |

Mixed-precision training via PyTorch `DataParallel`. Training time: 6–8 hours for U-Net.

---

## 🚀 Getting Started

### Requirements

```bash
pip install torch torchvision segmentation-models-pytorch albumentations kaggle
```

### Dataset

```bash
# Requires Kaggle API credentials
kaggle competitions download -c understanding_cloud_organization
```

### Run Training

```bash
jupyter notebook u2net.ipynb
```

The notebook covers the full pipeline: data loading → augmentation → model training → threshold optimization → inference.

---

## 📁 Repository Structure

```
cloud-pattern-segmentation/
├── u2net.ipynb       # Full pipeline: training, evaluation, inference
├── manuscript.pdf    # Research paper (Boğaziçi University)
└── LICENSE
```

---

## 📄 Citation

If you use this work, please cite:

```
Akdoğan, U.E., Alagöz, F. (2024). Deep Learning-Based Cloud Pattern Segmentation
from Satellite Imagery. Boğaziçi University.
```

---

## 📚 References

Core references from the paper:

- Rasp et al. (2019) — *Combining Crowd-Sourcing and Deep Learning to Explore Mesoscale Organization of Shallow Convection*, BAMS
- Ronneberger et al. (2015) — *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI
- Lin et al. (2017) — *Feature Pyramid Networks for Object Detection*, CVPR
- Chen et al. (2017) — *DeepLab: Semantic Image Segmentation with Atrous Convolution*, IEEE TPAMI

---

## 📝 License

Licensed under the Apache License 2.0. See [LICENSE](./LICENSE) for details.

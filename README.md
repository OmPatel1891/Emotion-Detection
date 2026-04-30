# EmotionClassNet: Emotion Recognition from Degraded Facial Images

> **DenseNet121-based emotion classification framework that handles low-definition, noisy, and watermarked facial images — tackling real-world data quality challenges head-on.**

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![IEEE](https://img.shields.io/badge/Published-IEEE%20AISTS%202025-blue)](https://ieee.org)

---

## Overview

Most emotion recognition models assume clean, high-resolution input. Real-world deployments rarely provide this — surveillance footage is grainy, web-scraped data is watermarked, and mobile captures are low-res. **EmotionClassNet** directly addresses this gap by training DenseNet121 on a challenging dataset of degraded facial images, producing a model that generalizes to conditions where standard classifiers fail.

**Published:** IEEE AISTS 2025 Conference

---

## Problem Statement

Classify human facial expressions into discrete emotion categories from images that may be:
- **Low-definition** (blurry, pixelated)
- **Watermarked** (text/logo overlays obscuring features)
- **Noisy** (compression artifacts, poor lighting)

---

## Emotion Classes

`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

---

## Architecture: DenseNet121

DenseNet121 was chosen for its dense connectivity pattern, which promotes feature reuse and gradient flow — particularly valuable when input quality is degraded and subtle facial feature signals are hard to extract.

Key design choices:
- ImageNet-pretrained DenseNet121 backbone (transfer learning)
- Custom classification head for emotion categories
- Dropout regularization to combat overfitting on noisy data
- Preprocessing pipeline tuned for low-quality inputs

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Framework | TensorFlow 2.x / Keras |
| Model | DenseNet121 (transfer learning) |
| Libraries | NumPy, OpenCV, Matplotlib, scikit-learn |
| Environment | Jupyter Notebook |

---

## Project Structure

```
Emotion-Detection/
└── EmotionClassNet.ipynb   # Full pipeline: data loading → preprocessing → training → evaluation
```

---

## Methodology

1. **Dataset** — Facial expression dataset with intentionally degraded images (watermarked + low-res variants)
2. **Preprocessing** — Grayscale conversion, histogram equalization, normalization, and face alignment
3. **Augmentation** — Random flips, rotation, zoom to improve generalization
4. **Training** — Fine-tuned DenseNet121 with Adam optimizer; categorical cross-entropy loss
5. **Evaluation** — Per-class precision, recall, F1-score; confusion matrix analysis

---

## Key Results

> Full metrics, confusion matrix, and per-class performance breakdown are in `EmotionClassNet.ipynb`.

The model demonstrates meaningful performance on degraded inputs where baseline CNNs show substantial degradation, validating the architecture choice for noisy real-world conditions.

---

## Real-World Applications

- Affective computing in human-computer interaction
- Customer sentiment analysis from video streams
- Accessibility tools for individuals with social communication challenges
- Mental health monitoring systems

---

## Publication

This work was published at **IEEE AISTS 2025** as **EmotionClassNet**.

---

## Author

**Om Patel** | MS Data Science, University of Michigan Ann Arbor  
[LinkedIn](https://www.linkedin.com/in/om-patel-20507a219/) · [GitHub](https://github.com/OmPatel1891)
